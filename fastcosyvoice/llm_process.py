"""LLM worker process for multi-process TTS.

Runs in a separate process so Flow+HiFT can run concurrently without GIL contention.

Message protocol
----------------
request_q (main → LLM):
  {'type': 'infer', 'batch_id': int,
   'text': Tensor, 'text_len': Tensor,
   'prompt_text': Tensor, 'prompt_text_len': Tensor,
   'llm_prompt_speech_token': Tensor, 'llm_prompt_speech_token_len': Tensor,
   'llm_embedding': Tensor, 'LM_latents': Tensor}
  {'type': 'stop'}

token_q (LLM → Flow):
  {'batch_id': int, 'tokens': List[int]}   # per-step tokens; negative = sample done
  {'batch_id': int, 'done': True}           # entire batch finished
"""

import logging
import os
import sys
import time
from typing import Optional


def _llm_worker_fn(
    model_dir: str,
    llm_pt_path: str,
    request_q,            # mp.Queue  main → LLM
    token_qs,             # List[mp.Queue]  LLM → each Flow proc
    init_q,               # mp.Queue  LLM  → main  (startup 'ready' signal)
    device_str: str = 'cuda:0',
):
    """Entry point for the LLM worker process (spawned)."""

    # ---- sys.path: child process (spawn) starts fresh --------------------
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _matcha = os.path.join(_root, 'third_party', 'Matcha-TTS')
    if _matcha not in sys.path:
        sys.path.insert(0, _matcha)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | [LLM-proc] %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger(__name__)

    import torch
    from hyperpyyaml import load_hyperpyyaml

    if torch.cuda.is_available():
        device = torch.device(device_str)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    logger.info('Loading LLM config from %s', model_dir)
    hf_llm_dir = os.path.join(model_dir, 'CosyVoice-BlankEN')
    hyper_yaml_path = os.path.join(model_dir, 'cosyvoice3_xvec.yaml')
    print('loading yaml from', hyper_yaml_path)
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': hf_llm_dir})

    llm = configs['llm']
    del configs

    cpt = torch.load(llm_pt_path, map_location='cpu')
    cpt.pop('epoch', None)
    cpt.pop('step', None)
    llm.load_state_dict(cpt, strict=False)
    del cpt
    llm.to(device)
    llm.eval()

    logger.info('LLM ready on %s', device)
    init_q.put({'type': 'ready', 'from': 'llm'})

    # ---- prefix cache store (spk_id → DynamicCache) ----------------------
    prefix_caches = {}

    # ---- main loop -------------------------------------------------------
    while True:
        msg = request_q.get()

        if msg['type'] == 'stop':
            logger.info('Stopping')
            break

        # ── Register speaker: compute & store prefix KV cache ─────────────
        if msg['type'] == 'register_spk':
            spk_id = msg['spk_id']
            with torch.inference_mode():
                cache = llm.compute_prefix_cache(
                    prompt_text=msg['prompt_text'].to(device),
                    prompt_text_len=msg['prompt_text_len'].to(device, dtype=torch.int32),
                    LM_latents=msg['LM_latents'].to(device),
                )
            prefix_caches[spk_id] = cache
            logger.info('Prefix cache stored for speaker: %s', spk_id)
            continue

        if msg['type'] != 'infer':
            logger.warning('Unknown message type: %s', msg['type'])
            continue

        batch_id: int     = msg['batch_id']
        token_splits: list = msg.get('token_splits', [msg['text'].size(0)])

        text        = msg['text'].to(device)
        text_len    = msg['text_len'].to(device, dtype=torch.int32)
        prompt_text     = msg['prompt_text'].to(device)
        prompt_text_len = msg['prompt_text_len'].to(device, dtype=torch.int32)
        prompt_speech_token     = msg['llm_prompt_speech_token'].to(device)
        prompt_speech_token_len = msg['llm_prompt_speech_token_len'].to(device, dtype=torch.int32)
        llm_embedding = msg['llm_embedding'].to(device)
        LM_latents    = msg['LM_latents'].to(device)

        # Look up prefix cache for this speaker
        spk_id = msg.get('spk_id')
        prefix_cache = prefix_caches.get(spk_id) if spk_id else None

        batch_size = text.size(0)
        token_count = 0

        # Build list of (queue, split_size) for active flow procs only
        active_qs = [(q, s) for q, s in zip(token_qs, token_splits) if s > 0]

        try:
            with torch.inference_mode():
                for token in llm.inference(
                    text=text,
                    text_len=text_len,
                    prompt_text=prompt_text,
                    prompt_text_len=prompt_text_len,
                    prompt_speech_token=prompt_speech_token,
                    prompt_speech_token_len=prompt_speech_token_len,
                    embedding=llm_embedding,
                    LM_latents=LM_latents,
                    prefix_cache=prefix_cache,
                ):
                    if torch.is_tensor(token):
                        step_tokens = token.detach().cpu().reshape(-1).tolist()
                    else:
                        step_tokens = [int(token)]

                    # Pad to batch_size (in case generator yields fewer)
                    while len(step_tokens) < batch_size:
                        step_tokens.append(-1)

                    for t in step_tokens:
                        if t is not None and int(t) >= 0:
                            token_count += 1

                    # Split step_tokens across active flow procs
                    offset = 0
                    for q, split_size in active_qs:
                        sub = step_tokens[offset:offset + split_size]
                        q.put({'batch_id': batch_id, 'tokens': sub})
                        offset += split_size

        except Exception as e:
            logger.error('Inference error: %s', e, exc_info=True)
        finally:
            for q, _ in active_qs:
                q.put({'batch_id': batch_id, 'done': True})
            logger.info('Batch %d done, total tokens=%d', batch_id, token_count)

            del text, text_len, prompt_text, prompt_text_len
            del prompt_speech_token, prompt_speech_token_len, llm_embedding, LM_latents

    logger.info('LLM process exiting')
