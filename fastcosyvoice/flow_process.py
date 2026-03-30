"""Flow + HiFT worker process for multi-process TTS.

Runs in a separate process so LLM can run concurrently without GIL contention.
Internally bridges multiprocessing queues to the existing threading-based
_flow_hift_job without modifying model.py.

Message protocol
----------------
setup_q (main → Flow):
  {'type': 'setup', 'batch_id': int, 'batch_size': int,
   'flow_prompt_speech_token': Tensor,  'flow_prompt_speech_token_len': Tensor,
   'prompt_speech_feat': Tensor,         'prompt_speech_feat_len': Tensor,
   'flow_embedding': Tensor,             'prompt_token_pad': Tensor}
  {'type': 'stop'}

token_q (LLM → Flow):
  {'batch_id': int, 'tokens': List[int]}   # per-step; negative = sample done
  {'batch_id': int, 'done': True}

audio_q (Flow → main):
  {'batch_id': int, 'sample_idx': int, 'tts_speech': Tensor}  # CPU tensor
  {'batch_id': int, 'done': True}
"""

import logging
import os
import queue
import sys
import threading
import time
import torch


class _DummyLLM(torch.nn.Module):
    """Placeholder so FastCosyVoice3Model constructor is satisfied."""
    def __init__(self):
        super().__init__()

    def inference(self, *args, **kwargs):
        raise RuntimeError('_DummyLLM.inference must never be called in the Flow process')


def _flow_worker_fn(
    model_dir: str,
    flow_pt_path: str,
    hift_pt_path: str,
    fp16: bool,
    load_trt: bool,
    flow_trt_max_batch_size: int,
    token_q,              # mp.Queue  LLM → Flow
    setup_q,              # mp.Queue  main → Flow
    audio_q,              # mp.Queue  Flow → main
    init_q,               # mp.Queue  Flow → main  (startup 'ready' signal)
    device_str: str = 'cuda:1',
    proc_idx: int = 0,
):
    """Entry point for the Flow+HiFT worker process (spawned)."""

    # ---- sys.path: child process (spawn) starts fresh --------------------
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _matcha = os.path.join(_root, 'third_party', 'Matcha-TTS')
    if _matcha not in sys.path:
        sys.path.insert(0, _matcha)

    tag = f'Flow-{proc_idx}({device_str})'
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s | %(levelname)s | [{tag}] %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger(__name__)

    from hyperpyyaml import load_hyperpyyaml
    from fastcosyvoice.model import FastCosyVoice3Model

    if torch.cuda.is_available():
        device = torch.device(device_str)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    logger.info('Loading Flow+HiFT config from %s', model_dir)
    hf_llm_dir = os.path.join(model_dir, 'CosyVoice-BlankEN')
    hyper_yaml_path = os.path.join(model_dir, 'cosyvoice3_xvec.yaml')
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': hf_llm_dir})
    flow = configs['flow']
    hift = configs['hift']
    del configs

    # Flow weights
    cpt = torch.load(flow_pt_path, map_location='cpu')
    cpt.pop('epoch', None)
    cpt.pop('step', None)
    flow.load_state_dict(cpt, strict=False)
    del cpt
    flow.to(device)
    if fp16:
        flow.half()
    flow.eval()

    # HiFT weights
    # raw_sd = torch.load(hift_pt_path, map_location=device, weights_only=True)
    # hift_sd = {k.removeprefix('generator.'): v for k, v in raw_sd.items()
    #            if k.startswith('generator.')}
    
    
    hift_state_dict = {
        k.replace('generator.', ''): v 
        for k, v in torch.load(hift_pt_path, map_location='cpu').items()
    }
    
    # raw_sd = torch.load(hift_pt_path, map_location=device, weights_only=True)
    # hift_state_dict = {k.removeprefix("generator."): v for k, v in raw_sd.items()
    #                 if k.startswith("generator.")}
    
    hift.load_state_dict(hift_state_dict, strict=True)
    hift.to(device).eval()
    # del hift_state_dict

    # Build model shell with dummy LLM (_flow_hift_job never calls self.llm)
    model = FastCosyVoice3Model(_DummyLLM(), flow, hift, fp16=fp16)

    # Load TRT engine for Flow decoder
    if load_trt:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(device)
            sm_version = f'sm{major}{minor}'
        else:
            sm_version = 'cpu'

        if model.use_flow_cache:
            # Cache flow UNet decoder (CausalConditionalDecoder with KV cache)
            trt_plan = os.path.join(
                model_dir,
                f'flow_cache.decoder.estimator.{"fp16" if fp16 else "fp32"}.b{flow_trt_max_batch_size}.{sm_version}.plan'
            )
            onnx_path = os.path.join(
                model_dir,
                'flow_cache.decoder.estimator.fp32.onnx'
            )
            model.load_trt_cache(trt_plan, onnx_path, 1, fp16, max_batch_size=flow_trt_max_batch_size)
            logger.info('Cache flow TRT loaded (max_batch=%d): %s', flow_trt_max_batch_size, trt_plan)
        else:
            # DiT decoder (stateless)
            trt_plan = os.path.join(
                model_dir,
                f'flow.decoder.estimator.{"fp16" if fp16 else "fp32"}.b{flow_trt_max_batch_size}.{sm_version}.plan'
            )
            onnx_path = os.path.join(
                model_dir,
                f'flow.decoder.estimator.fp32.b{flow_trt_max_batch_size}.onnx'
            )
            model.load_trt(trt_plan, onnx_path, 1, fp16, max_batch_size=flow_trt_max_batch_size)
            logger.info('DiT TRT loaded: %s', trt_plan)

    logger.info('Flow+HiFT ready on %s', device)
    init_q.put({'type': 'ready', 'from': tag})

    # ---- main loop -------------------------------------------------------
    while True:
        setup_msg = setup_q.get()

        if setup_msg['type'] == 'stop':
            logger.info('Stopping')
            break

        if setup_msg['type'] != 'setup':
            logger.warning('Unknown setup message: %s', setup_msg['type'])
            continue

        batch_id: int = setup_msg['batch_id']
        batch_size: int = setup_msg['batch_size']
        dtype = torch.float16 if fp16 else torch.float32

        flow_prompt_token_gpu     = setup_msg['flow_prompt_speech_token'].to(device)
        flow_prompt_token_len_gpu = setup_msg['flow_prompt_speech_token_len'].to(device, dtype=torch.int32)
        prompt_feat_gpu           = setup_msg['prompt_speech_feat'].to(device, dtype=dtype)
        prompt_feat_len_gpu       = setup_msg['prompt_speech_feat_len'].to(device, dtype=torch.int32)
        flow_embedding_gpu        = setup_msg['flow_embedding'].to(device, dtype=dtype)
        prompt_token_pad          = setup_msg['prompt_token_pad'].to(device, dtype=torch.int32)

        # Shared threading state (consumed by _flow_hift_job)
        tokens_list: list        = [[] for _ in range(batch_size)]
        tokens_lock               = threading.Lock()
        llm_end_flag: dict        = {'done': False}
        llm_sample_done: list     = [False] * batch_size

        # Threading queue: _flow_hift_job → audio_forwarder
        thread_audio_q: queue.Queue = queue.Queue(maxsize=8)

        # ------------------------------------------------------------------
        # token_feeder: mp.Queue (token_q) → threading shared state
        # ------------------------------------------------------------------
        def token_feeder(bid=batch_id, bs=batch_size):
            while True:
                msg = token_q.get()
                if msg['batch_id'] != bid:
                    # Unexpected; put back and wait briefly
                    token_q.put(msg)
                    time.sleep(0.001)
                    continue

                if msg.get('done'):
                    with tokens_lock:
                        for i in range(bs):
                            if not llm_sample_done[i]:
                                llm_sample_done[i] = True
                        llm_end_flag['done'] = True
                    break

                step_tokens = msg['tokens']
                with tokens_lock:
                    for sample_idx, t in enumerate(step_tokens):
                        if sample_idx >= bs:
                            break
                        if t is None or int(t) < 0:
                            llm_sample_done[sample_idx] = True
                        else:
                            tokens_list[sample_idx].append(int(t))

        # ------------------------------------------------------------------
        # audio_forwarder: threading.Queue → mp.Queue (audio_q)
        # ------------------------------------------------------------------
        def audio_forwarder(bid=batch_id):
            while True:
                chunk = thread_audio_q.get()
                if chunk is None:  # sentinel from _flow_hift_job
                    audio_q.put({'batch_id': bid, 'done': True})
                    break

                ready = chunk.pop('_ready_event', None)
                if ready is not None:
                    ready.synchronize()

                audio_tensor = chunk['tts_speech']
                sample_idx = int(chunk.get('sample_idx', 0))
                is_finalize = chunk.get('finalize', False)
                # Convert to regular CPU tensor for pickle serialization
                audio_cpu = audio_tensor.detach().cpu().clone()
                audio_q.put({
                    'batch_id': bid,
                    'sample_idx': sample_idx,
                    'tts_speech': audio_cpu,
                    'finalize': is_finalize,
                })

        feeder_t   = threading.Thread(target=token_feeder,   daemon=True)
        forwarder_t = threading.Thread(target=audio_forwarder, daemon=True)
        feeder_t.start()
        forwarder_t.start()

        # Run the existing Flow+HiFT pipeline (blocks until done)
        model._flow_hift_job(
            tokens_list=tokens_list,
            tokens_lock=tokens_lock,
            llm_end_flag=llm_end_flag,
            audio_queue=thread_audio_q,
            flow_prompt_token_gpu=flow_prompt_token_gpu,
            flow_prompt_token_len_gpu=flow_prompt_token_len_gpu,
            prompt_feat_gpu=prompt_feat_gpu,
            prompt_feat_len_gpu=prompt_feat_len_gpu,
            flow_embedding_gpu=flow_embedding_gpu,
            prompt_token_pad=prompt_token_pad,
            llm_sample_done=llm_sample_done,
        )

        feeder_t.join(timeout=5.0)
        forwarder_t.join(timeout=5.0)

        logger.info('Batch %d complete', batch_id)

        del flow_prompt_token_gpu, flow_prompt_token_len_gpu
        del prompt_feat_gpu, prompt_feat_len_gpu, flow_embedding_gpu, prompt_token_pad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info('Flow process exiting')
