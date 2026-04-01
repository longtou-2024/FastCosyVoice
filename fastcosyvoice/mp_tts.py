"""Multi-process TTS orchestrator.

Architecture (multi-flow example: 2 Flow procs)
-------------------------------------------------
                    ┌──────────────────────┐
                    │    Main Process      │
                    │  (frontend + this)   │
                    └──┬────────┬──┬───────┘
              request_q│  setup_q0│ │setup_q1
                       ▼         ▼ ▼
             ┌──────────────┐  ┌──────┐ ┌──────┐
             │  LLM Process │  │Flow 0│ │Flow 1│
             │  (batch=N)   │  │(b=8) │ │(b=8) │
             └──┬───────────┘  └──┬───┘ └──┬───┘
          token_q0───────────────►│   │    │
          token_q1───────────────────►│    │
                                 audio_q0  audio_q1
                                      ↘  ↙
                               Main Process (yield PCM)

token_splits=[8,8] sent in each LLM request so the LLM worker routes
step_tokens[0:8] → token_q0 and step_tokens[8:16] → token_q1.
"""

import logging
import os
import queue as tqueue
import threading
import time
from typing import Generator, List, Optional

import torch
import torch.multiprocessing as mp
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from cosyvoice.utils.frontend_utils import contains_cyrillic, convert_stress_marks, split_text_smart

from .frontend import CosyVoiceFrontEnd
from .llm_process import _llm_worker_fn
from .flow_process import _flow_worker_fn

logger = logging.getLogger(__name__)

_TOKEN_HOP_LEN = 25   # must match FastCosyVoice3Model.token_hop_len


def _compute_splits(total: int, num_parts: int) -> List[int]:
    """Distribute `total` samples across `num_parts` parts as evenly as possible."""
    base, rem = divmod(total, num_parts)
    return [base + (1 if i < rem else 0) for i in range(num_parts)]


class MultiProcessTTS:
    """
    Drop-in replacement for FastCosyVoice3 that runs LLM and Flow+HiFT in
    separate processes to bypass the Python GIL.

    Supports multiple Flow+HiFT processes to split a large batch across GPUs
    or to pipeline work on the same GPU.

    Parameters
    ----------
    flow_devices : list of str
        One device string per Flow process.  The number of Flow processes is
        ``len(flow_devices)``.  Defaults to ``['cuda:1', 'cuda:1']`` (two Flow
        processes both on GPU 1).  Use ``['cuda:0']`` for a single-GPU setup.

    Supported interface (same as FastCosyVoice3):
      - add_zero_shot_spk(prompt_text, prompt_wav, spk_id)
      - inference_zero_shot_stream(tts_text, prompt_text, prompt_wav, caption,
                                   zero_shot_spk_id='', ...)
      - sample_rate  (property)
    """

    def __init__(
        self,
        model_dir: str,
        llm_pt_path: str,
        flow_pt_path: str,
        hift_pt_path: str,
        fp16: bool = True,
        load_trt: bool = True,
        flow_trt_max_batch_size: int = 8,
        llm_device: str = 'cuda:0',
        flow_devices: Optional[List[str]] = None,
        qwen3_dir: str = None,
        on_frontend_loaded: Optional[callable] = None,
        on_workers_ready: Optional[callable] = None,
    ):
        self.model_dir = model_dir
        self.fp16 = fp16
        self._batch_id = 0

        if flow_devices is None:
            flow_devices = ['cuda:1', 'cuda:1']
        self._num_flow_procs = len(flow_devices)

        # LM_latents: fixed caption embedding (same as cosyvoice.py)
        # self.LM_latents = torch.load('LM_latents_happy.pt')  # CPU tensor

        # ── Load frontend (stays in main process) ────────────────────────
        hf_llm_dir = os.path.join(model_dir, 'CosyVoice-BlankEN')
        hyper_yaml_path = os.path.join(model_dir, 'cosyvoice3_xvec.yaml')
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': hf_llm_dir})

        self.sample_rate: int = configs['sample_rate']
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            os.path.join(model_dir, 'campplus.onnx'),
            os.path.join(model_dir, 'speech_tokenizer_v3.onnx'),
            os.path.join(model_dir, 'spk2info.pt'),
            configs['allowed_special'],
            qwen3_model_dir=qwen3_dir,
        )
        del configs

        # Callback: frontend loaded, Qwen3/ONNX files no longer needed on disk
        if on_frontend_loaded:
            on_frontend_loaded()

        # ── Create inter-process queues ───────────────────────────────────
        ctx = mp.get_context('spawn')
        self._request_q = ctx.Queue(maxsize=4)          # main → LLM
        self._init_q    = ctx.Queue()                    # workers → main (ready)

        # Per-flow-proc queues
        self._token_qs = [ctx.Queue(maxsize=1024) for _ in range(self._num_flow_procs)]
        self._setup_qs = [ctx.Queue(maxsize=4)    for _ in range(self._num_flow_procs)]
        self._audio_qs = [ctx.Queue(maxsize=64)   for _ in range(self._num_flow_procs)]

        # ── Start LLM process (receives list of token_qs) ─────────────────
        self._llm_proc = ctx.Process(
            target=_llm_worker_fn,
            args=(model_dir, llm_pt_path,
                  self._request_q, self._token_qs, self._init_q,
                  llm_device),
            daemon=True,
        )
        self._llm_proc.start()

        # ── Start Flow+HiFT processes ─────────────────────────────────────
        self._flow_procs = []
        for proc_idx, (token_q, setup_q, audio_q, flow_dev) in enumerate(zip(
                self._token_qs, self._setup_qs, self._audio_qs, flow_devices)):
            p = ctx.Process(
                target=_flow_worker_fn,
                args=(model_dir, flow_pt_path, hift_pt_path,
                      fp16, load_trt, flow_trt_max_batch_size,
                      token_q, setup_q, audio_q, self._init_q,
                      flow_dev, proc_idx),
                daemon=True,
            )
            p.start()
            self._flow_procs.append(p)

        # Wait for all workers to signal readiness (1 LLM + N Flow procs)
        total_workers = 1 + self._num_flow_procs
        logger.info('Waiting for %d worker processes to initialise...', total_workers)
        ready_count = 0
        while ready_count < total_workers:
            msg = self._init_q.get(timeout=600)
            ready_count += 1
            logger.info('%s process ready (%d/%d)', msg['from'], ready_count, total_workers)
        logger.info('All worker processes ready')

        # Callback: all models loaded into GPU, disk files no longer needed
        if on_workers_ready:
            on_workers_ready()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _next_batch_id(self) -> int:
        self._batch_id += 1
        return self._batch_id

    @staticmethod
    def _resolve_lengths(lengths, tensor):
        if lengths is None:
            lengths = torch.full((tensor.size(0),), tensor.size(1), dtype=torch.int32)
        if lengths.dim() == 0:
            lengths = lengths.reshape(1)
        return lengths.to(dtype=torch.int32)

    @staticmethod
    def _expand_batch(tensor, batch_size):
        if tensor.size(0) == batch_size:
            return tensor
        if tensor.size(0) == 1:
            return tensor.expand(batch_size, *[-1] * (tensor.dim() - 1))
        return tensor

    @staticmethod
    def _tensor_to_pcm_bytes(audio_tensor: torch.Tensor) -> bytes:
        audio = audio_tensor.squeeze().clamp(-1.0, 1.0)
        return (audio * 32767).to(torch.int16).cpu().numpy().tobytes()

    def _process_stress(self, text: str, auto_stress: bool = False) -> str:
        if auto_stress and contains_cyrillic(text):
            if not hasattr(self, '_accentor'):
                from silero_stress import load_accentor
                self._accentor = load_accentor()
                self._accentor.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            chunks = split_text_smart(text, max_chars=400)
            text = ' '.join(self._accentor(c, stress_single_vowel=False) for c in chunks)
        return convert_stress_marks(text)

    # ── public API ───────────────────────────────────────────────────────────

    def add_zero_shot_spk(
        self,
        prompt_text: str,
        prompt_wav: str,
        zero_shot_spk_id: str,
    ) -> bool:
        """Register a zero-shot speaker (same interface as FastCosyVoice3)."""
        if zero_shot_spk_id == '':
            raise ValueError('zero_shot_spk_id cannot be empty')
        caption = '"약하게" "기쁜" 감정이고 "독백체" 스타일<|endofprompt|>'
        model_input = self.frontend.frontend_zero_shot(
            caption, '', prompt_text, prompt_wav, self.sample_rate, ''
        )
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input

        # Ask LLM process to pre-compute prefix KV cache for this speaker
        self._request_q.put({
            'type': 'register_spk',
            'spk_id': zero_shot_spk_id,
            'prompt_text': model_input['prompt_text'].clone(),
            'prompt_text_len': model_input['prompt_text_len'].clone()
                if torch.is_tensor(model_input['prompt_text_len'])
                else torch.tensor([model_input['prompt_text_len']], dtype=torch.int32),
            'LM_latents': model_input['LM_latents'].clone(),
        })
        return True

    def inference_zero_shot_stream(
        self,
        tts_text,
        prompt_text: str = '',
        prompt_wav: str = '',
        caption: str = '',
        zero_shot_spk_id: str = '',
        zero_shot_spk_ids: Optional[List[str]] = None,
        text_frontend: bool = True,
        auto_stress: bool = False,
    ) -> Generator:
        """
        Zero-shot streaming TTS via multi-process LLM + Flow/HiFT.

        Yields raw PCM bytes (int16, little-endian, mono) for single text,
        or dicts {'sample_idx': int, 'pcm_bytes': bytes} for batch text.

        For multi-speaker batch: pass ``zero_shot_spk_ids`` (one per tts_text
        element).  Each speaker must be pre-registered via add_zero_shot_spk().
        """
        if prompt_text:
            prompt_text = self.frontend.text_normalize(
                prompt_text, split=False, text_frontend=text_frontend
            )

        if isinstance(tts_text, (list, tuple)):
            text_batches = [[
                self.frontend.text_normalize(
                    self._process_stress(t, auto_stress),
                    split=False, text_frontend=text_frontend,
                )
                for t in tts_text
            ]]
        else:
            processed = self._process_stress(tts_text, auto_stress)
            text_batches = [[chunk] for chunk in tqdm(
                self.frontend.text_normalize(processed, split=True, text_frontend=text_frontend),
                desc='Synthesising',
            )]

        for text_batch in text_batches:
            # ── prepare model inputs ──────────────────────────────────────
            model_inputs = []

            if zero_shot_spk_ids is not None:
                # ── Multi-speaker: per-text speaker ID ────────────────────
                if len(zero_shot_spk_ids) != len(text_batch):
                    raise ValueError(
                        f'zero_shot_spk_ids length ({len(zero_shot_spk_ids)}) '
                        f'!= tts_text length ({len(text_batch)})'
                    )
                for text_item, sid in zip(text_batch, zero_shot_spk_ids):
                    if sid not in self.frontend.spk2info:
                        raise ValueError(f'Unknown zero_shot_spk_id: {sid}')
                    text_token, text_token_len = self.frontend._extract_text_token(text_item)
                    inp = {k: (v.clone() if torch.is_tensor(v) else v)
                           for k, v in self.frontend.spk2info[sid].items()}
                    inp['text']     = text_token
                    inp['text_len'] = text_token_len
                    model_inputs.append(inp)
            elif zero_shot_spk_id:
                # ── Single-speaker (기존 동작) ────────────────────────────
                if zero_shot_spk_id not in self.frontend.spk2info:
                    raise ValueError(f'Unknown zero_shot_spk_id: {zero_shot_spk_id}')
                for text_item in text_batch:
                    text_token, text_token_len = self.frontend._extract_text_token(text_item)
                    inp = {k: (v.clone() if torch.is_tensor(v) else v)
                           for k, v in self.frontend.spk2info[zero_shot_spk_id].items()}
                    inp['text']     = text_token
                    inp['text_len'] = text_token_len
                    model_inputs.append(inp)
            else:
                model_inputs = [
                    self.frontend.frontend_zero_shot(
                        caption, text_item, prompt_text, prompt_wav,
                        self.sample_rate, '',
                    )
                    for text_item in text_batch
                ]

            # Stack batch
            stacked: dict = {}
            for key in model_inputs[0].keys():
                values = [item[key] for item in model_inputs]
                if not torch.is_tensor(values[0]):
                    stacked[key] = values
                    continue
                if key.endswith('_len'):
                    stacked[key] = torch.cat([v.reshape(-1) for v in values], dim=0)
                    continue
                normalized = [v.squeeze(0) if v.dim() > 0 and v.size(0) == 1 else v
                              for v in values]
                pad_val = 0.0 if normalized[0].dtype.is_floating_point else 0
                if normalized[0].dim() == 0:
                    stacked[key] = torch.stack(normalized, dim=0)
                else:
                    stacked[key] = torch.nn.utils.rnn.pad_sequence(
                        normalized, batch_first=True, padding_value=pad_val,
                    )

            batch_size = len(text_batch)

            # Resolve lengths and expand shared tensors to full batch_size

            text              = stacked['text']
            text_len          = self._resolve_lengths(stacked.get('text_len'), text)
            prompt_text_t     = self._expand_batch(stacked['prompt_text'], batch_size)
            prompt_text_len   = self._resolve_lengths(stacked.get('prompt_text_len'), prompt_text_t)
            llm_pst           = self._expand_batch(stacked['llm_prompt_speech_token'], batch_size)
            llm_pst_len       = self._resolve_lengths(stacked.get('llm_prompt_speech_token_len'), llm_pst)
            flow_pst          = self._expand_batch(stacked['flow_prompt_speech_token'], batch_size)
            flow_pst_len      = self._resolve_lengths(stacked.get('flow_prompt_speech_token_len'), flow_pst)
            prompt_feat       = self._expand_batch(stacked['prompt_speech_feat'], batch_size)
            prompt_feat_len   = self._resolve_lengths(stacked.get('prompt_speech_feat_len'), prompt_feat)
            flow_emb          = self._expand_batch(stacked['flow_embedding'], batch_size)
            llm_emb           = stacked['llm_embedding']
            if llm_emb.numel() != 0:
                llm_emb = self._expand_batch(llm_emb, batch_size)

            # LM_latents: tile the pre-computed fixed latent
            # lm_latents = caption.tile(batch_size, 1, 1)
            lm_latents = stacked['LM_latents']
            # prompt_token_pad (per sample)
            prompt_token_pad = (
                ((flow_pst_len + _TOKEN_HOP_LEN - 1) // _TOKEN_HOP_LEN) * _TOKEN_HOP_LEN
                - flow_pst_len
            ).to(dtype=torch.int32)

            # ── split batch across flow procs ─────────────────────────────
            token_splits = _compute_splits(batch_size, self._num_flow_procs)
            # active procs: (proc_idx, split_size, sample_offset)
            active_procs = []
            offset = 0
            for i, split_size in enumerate(token_splits):
                if split_size > 0:
                    active_procs.append((i, split_size, offset))
                offset += split_size

            batch_id   = self._next_batch_id()
            start_time = time.time()

            # ── send LLM request (full batch + routing info) ──────────────
            self._request_q.put({
                'type': 'infer',
                'batch_id': batch_id,
                'spk_id': zero_shot_spk_id if (zero_shot_spk_id and zero_shot_spk_ids is None) else None,
                'token_splits':                token_splits,
                'text':                        text,
                'text_len':                    text_len,
                'prompt_text':                 prompt_text_t,
                'prompt_text_len':             prompt_text_len,
                'llm_prompt_speech_token':     llm_pst,
                'llm_prompt_speech_token_len': llm_pst_len,
                'llm_embedding':               llm_emb,
                'LM_latents':                  lm_latents,
            })

            # ── send setup to each active flow proc (sliced sub-batch) ────
            for proc_idx, split_size, sub_start in active_procs:
                sub_end = sub_start + split_size
                self._setup_qs[proc_idx].put({
                    'type': 'setup',
                    'batch_id': batch_id,
                    'batch_size': split_size,
                    'flow_prompt_speech_token':
                        flow_pst[sub_start:sub_end].contiguous(),
                    'flow_prompt_speech_token_len':
                        flow_pst_len[sub_start:sub_end].contiguous(),
                    'prompt_speech_feat':
                        prompt_feat[sub_start:sub_end].contiguous(),
                    'prompt_speech_feat_len':
                        prompt_feat_len[sub_start:sub_end].contiguous(),
                    'flow_embedding':
                        flow_emb[sub_start:sub_end].contiguous(),
                    'prompt_token_pad':
                        prompt_token_pad[sub_start:sub_end].contiguous(),
                })

            # ── collect audio from all active flow procs via threads ───────
            merged_q: tqueue.Queue = tqueue.Queue()

            def _collect(proc_idx: int, sample_offset: int, aq, bid=batch_id):
                while True:
                    try:
                        chunk = aq.get(timeout=30.0)
                    except Exception:
                        logger.warning('Timeout waiting for audio from flow proc %d (batch %d)',
                                       proc_idx, bid)
                        merged_q.put(('error', proc_idx))
                        return
                    if chunk.get('done'):
                        merged_q.put(('done', proc_idx))
                        return
                    if chunk['batch_id'] != bid:
                        logger.warning('Unexpected batch_id from flow proc %d: %s vs %s',
                                       proc_idx, chunk['batch_id'], bid)
                        continue
                    # Offset sample_idx so it's global + tag with proc_idx
                    c = dict(chunk)
                    c['sample_idx'] = int(c.get('sample_idx', 0)) + sample_offset
                    c['flow_proc'] = proc_idx
                    merged_q.put(('audio', c))

            collector_threads = []
            for proc_idx, split_size, sample_offset in active_procs:
                t = threading.Thread(
                    target=_collect,
                    args=(proc_idx, sample_offset, self._audio_qs[proc_idx]),
                    daemon=True,
                )
                t.start()
                collector_threads.append(t)

            done_count = 0
            num_active = len(active_procs)
            while done_count < num_active:
                try:
                    kind, data = merged_q.get(timeout=60.0)
                except Exception:
                    logger.warning('Timeout collecting merged audio (batch %d)', batch_id)
                    break

                if kind in ('done', 'error'):
                    done_count += 1
                    continue

                # kind == 'audio'
                audio_tensor = data['tts_speech']
                sample_idx   = int(data.get('sample_idx', 0))
                speech_len   = audio_tensor.shape[-1] / self.sample_rate
                elapsed      = time.time() - start_time
                rtf = elapsed / speech_len if speech_len > 0 else 0.0
                flow_proc = data.get('flow_proc', '?')
                is_final = data.get('finalize', False)
                logger.info('Yield len=%.3fs rtf=%.3f sample_idx=%d, Flow_process=%s, finalize=%s', speech_len, rtf, sample_idx, flow_proc, is_final)

                pcm = self._tensor_to_pcm_bytes(audio_tensor)
                if batch_size == 1:
                    yield pcm
                else:
                    yield {'sample_idx': sample_idx, 'pcm_bytes': pcm}

                start_time = time.time()

            for t in collector_threads:
                t.join(timeout=5.0)

    def stop(self):
        """Gracefully shut down all worker processes."""
        try:
            self._request_q.put({'type': 'stop'})
        except Exception:
            pass
        for setup_q in self._setup_qs:
            try:
                setup_q.put({'type': 'stop'})
            except Exception:
                pass
        self._llm_proc.join(timeout=5.0)
        for p in self._flow_procs:
            p.join(timeout=5.0)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
