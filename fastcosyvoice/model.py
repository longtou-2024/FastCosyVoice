# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 FastCosyVoice Implementation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FastCosyVoice3Model - True Pipeline TTS Model

Architecture for maximum LLM throughput:
- LLM runs in dedicated thread with its own CUDA stream
- Flow+Hift run in separate thread (doesn't block LLM)  
- Main thread only yields audio from queue

This achieves 80-90% of isolated LLM throughput by never blocking LLM generation.
"""

import fcntl
import os
import queue
import threading
import time
from contextlib import nullcontext
from typing import Generator, Dict, List, Optional

import torch
from torch.nn import functional as F

from cosyvoice.utils.file_utils import logging, convert_onnx_to_trt
from cosyvoice.utils.common import TrtContextWrapper


class FastCosyVoice3Model:
    """
    True pipeline TTS model with non-blocking LLM.
    
    Pipeline architecture:
    [LLM Thread] → token_queue → [Flow+Hift Thread] → audio_queue → [Main Thread: yield]
    Key insight: Flow+Hift run in their own thread, so their blocking operations
    (TensorRT sync, Hift CPU f0_predictor) don't affect LLM at all.
    """
    
    def __init__(
        self,
        llm: torch.nn.Module,
        flow: torch.nn.Module,
        hift: torch.nn.Module,
        fp16: bool = False
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16

        # Token hop length must match training static_chunk_size
        self.token_hop_len = 25

        # ── Cache flow config (CausalMaskedDiffWithXvec) ──
        # Auto-detect: XVec has encoder.forward_chunk for streaming cache
        self.use_flow_cache = hasattr(self.flow, 'encoder') and hasattr(self.flow.encoder, 'forward_chunk')
        self.flow_n_timesteps = 3
        self.flow_decoder_required_cache_size = (
            1 * self.token_hop_len * getattr(self.flow, 'token_mel_ratio', 2)
            if self.use_flow_cache else 0
        )
        if self.use_flow_cache:
            logging.info(f'[FastCosyVoice3Model] Cache flow enabled: n_timesteps={self.flow_n_timesteps}, '
                         f'cache_size={self.flow_decoder_required_cache_size}')

        # LLM gets dedicated CUDA stream for true parallelism
        if torch.cuda.is_available():
            self.llm_stream = torch.cuda.Stream(self.device)
        else:
            self.llm_stream = None
    
    def _init_flow_cache(self, batch_size: int = 1):
        """Create flow cache for CausalMaskedDiffWithXvec streaming (batch supported).

        - encoder_caches: list of B per-sample encoder caches (forward_chunk KV cache)
        - decoder_cache: UNet decoder cache with CFG batch dim = 2*B
        """
        dev = self.device
        n_t = self.flow_n_timesteps
        cs = self.flow_decoder_required_cache_size
        B2 = 2 * batch_size  # CFG doubling
        dtype = torch.float16 if self.fp16 else torch.float32

        encoder_caches = []
        for _ in range(batch_size):
            ec = {
                "offset": 0,
                "pre_lookahead_layer_conv2_cache": torch.zeros(1, 512, 2, device=dev, dtype=dtype),
                "encoders_kv_cache": torch.zeros(6, 1, 8, 0, 64 * 2, device=dev, dtype=dtype),
                "upsample_offset": 0,
                "upsample_conv_cache": torch.zeros(1, 512, 4, device=dev, dtype=dtype),
                "upsample_kv_cache": torch.zeros(4, 1, 8, 0, 64 * 2, device=dev, dtype=dtype),
            }
            encoder_caches.append(ec)

        decoder_cache = {
            "offset": 0,
            "down_blocks_conv_cache": torch.zeros(n_t, 1, B2, 832, 2, device=dev, dtype=dtype),
            "down_blocks_kv_cache": torch.zeros(n_t, 1, 4, B2, cs, 512, 2, device=dev, dtype=dtype),
            "mid_blocks_conv_cache": torch.zeros(n_t, 12, B2, 512, 2, device=dev, dtype=dtype),
            "mid_blocks_kv_cache": torch.zeros(n_t, 12, 4, B2, cs, 512, 2, device=dev, dtype=dtype),
            "up_blocks_conv_cache": torch.zeros(n_t, 1, B2, 1024, 2, device=dev, dtype=dtype),
            "up_blocks_kv_cache": torch.zeros(n_t, 1, 4, B2, cs, 512, 2, device=dev, dtype=dtype),
            "final_blocks_conv_cache": torch.zeros(n_t, B2, 256, 2, device=dev, dtype=dtype),
        }

        return {"encoder_caches": encoder_caches, "decoder_cache": decoder_cache}

    # ── decoder_cache batch dimension map ────────────────────────────────
    # key → B2 dim (CFG-doubled: [cond_0..cond_N-1, uncond_0..uncond_N-1])
    _DECODER_CACHE_B2_DIM = {
        "down_blocks_conv_cache":  2,   # [n_t, 1,  B2, 832,  2]
        "down_blocks_kv_cache":    3,   # [n_t, 1,  4,  B2, cs, 512, 2]
        "mid_blocks_conv_cache":   2,   # [n_t, 12, B2, 512,  2]
        "mid_blocks_kv_cache":     3,   # [n_t, 12, 4,  B2, cs, 512, 2]
        "up_blocks_conv_cache":    2,   # [n_t, 1,  B2, 1024, 2]
        "up_blocks_kv_cache":      3,   # [n_t, 1,  4,  B2, cs, 512, 2]
        "final_blocks_conv_cache": 1,   # [n_t, B2, 256, 2]
    }

    def _merge_flow_caches(self, flow_caches: list, sample_indices: List[int]) -> dict:
        """Merge per-sample flow caches (each batch=1) into a single batched cache.

        Decoder expects B2 ordering: [cond_0, cond_1, ..., uncond_0, uncond_1, ...]
        Per-sample cache has B2=2:   [cond_i, uncond_i]
        """
        merged_enc = []
        for idx in sample_indices:
            merged_enc.extend(flow_caches[idx]['encoder_caches'])

        first_dec = flow_caches[sample_indices[0]]['decoder_cache']
        merged_dec = {'offset': first_dec['offset']}
        for key, b2_dim in self._DECODER_CACHE_B2_DIM.items():
            if key in first_dec:
                # Separate cond (index 0) and uncond (index 1) from each sample,
                # then concat as [all_conds..., all_unconds...]
                conds = [flow_caches[idx]['decoder_cache'][key].narrow(b2_dim, 0, 1) for idx in sample_indices]
                unconds = [flow_caches[idx]['decoder_cache'][key].narrow(b2_dim, 1, 1) for idx in sample_indices]
                merged_dec[key] = torch.cat(conds + unconds, dim=b2_dim)

        return {'encoder_caches': merged_enc, 'decoder_cache': merged_dec}

    def _split_flow_caches(self, merged_cache: dict, flow_caches: list, sample_indices: List[int]):
        """Split batched cache back into per-sample flow caches (batch=1 each).

        Decoder B2 ordering: [cond_0, cond_1, ..., uncond_0, uncond_1, ...]
        Need to reconstruct per-sample: [cond_i, uncond_i]
        """
        N = len(sample_indices)
        updated_enc = merged_cache['encoder_caches']
        updated_dec = merged_cache['decoder_cache']

        for i, sample_idx in enumerate(sample_indices):
            flow_caches[sample_idx]['encoder_caches'] = [updated_enc[i]]
            flow_caches[sample_idx]['decoder_cache'] = {'offset': updated_dec['offset']}
            for key, b2_dim in self._DECODER_CACHE_B2_DIM.items():
                if key in updated_dec:
                    cond_slice = updated_dec[key].narrow(b2_dim, i, 1)
                    uncond_slice = updated_dec[key].narrow(b2_dim, N + i, 1)
                    flow_caches[sample_idx]['decoder_cache'][key] = (
                        torch.cat([cond_slice, uncond_slice], dim=b2_dim).contiguous()
                    )

    def load(self, llm_model: str, flow_model: str, hift_model: str, *, load_llm: bool = True):
        """Load model weights from files.

        Args:
            llm_model: Path to PyTorch LLM weights (.pt)
            flow_model: Path to Flow weights (.pt)
            hift_model: Path to HiFT weights (.pt)
            load_llm: If False, skip loading/moving the PyTorch LLM to GPU.
                Useful when using TRT-LLM (LLM runs outside PyTorch), to reduce VRAM usage.
        """
        if load_llm:
            cpt=torch.load(llm_model, map_location='cpu')
            cpt.pop('epoch',None)
            cpt.pop('step',None)
            # print(cpt.keys())
            ### remap
            # from collections import OrderedDict

            # new_state_dict = OrderedDict()
            
            # for k, v in cpt.items():
            #     # llm.model.xxx -> llm.model.model.xxx
            #     # if k.startswith("llm.model.lm_head"):
            #     #     logging.warning(f"Skipping missing key: {k}")
            #     #     continue
            #     if k.startswith("llm.model."):
            #         k = k.replace("llm.model.", "llm.model.model.", 1)
            #     # if k.endswith("lm_head.weight") or ".lm_head." in k:
            #     #     continue
            #     new_state_dict[k] = v
            # print(new_state_dict)
            self.llm.load_state_dict(cpt, strict=False)
            self.llm.to(self.device)
            # if self.fp16:
            #     self.llm.half()
            self.llm.eval()

        cpt=torch.load(flow_model, map_location='cpu')
        cpt.pop('epoch',None)
        cpt.pop('step',None)
        
        self.flow.load_state_dict(cpt, strict=False)
        self.flow.to(self.device)
        # if self.fp16:
        #     self.flow.half()
        self.flow.eval()
        
        hift_state_dict = {
            k.replace('generator.', ''): v 
            for k, v in torch.load(hift_model, map_location='cpu').items()
        }
        
        raw_sd = torch.load(hift_model, map_location=self.device, weights_only=True)
        hift_state_dict = {k.removeprefix("generator."): v for k, v in raw_sd.items()
                        if k.startswith("generator.")}


        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
    
    def get_trt_kwargs(self, fp16: bool = True, max_batch_size: int = 1):
        """Get TensorRT optimization profiles for Flow decoder."""
        # NOTE: max_shape must be large enough to handle prompt_feat + generated mel
        # For zero-shot with long reference audio (~30s) + long output (~60s), need ~6000 frames
        # Must match estimator call signature and TensorRT bindings order in
        # `cosyvoice/flow/flow_matching.py::forward_estimator`.
        cfg_min_batch = 2
        cfg_max_batch = max(2, int(max_batch_size) * 2)
        cfg_opt_batch = cfg_max_batch
        min_shape = [(cfg_min_batch, 80, 4), (cfg_min_batch, 1, 4), (cfg_min_batch, 80, 4), (cfg_min_batch,), (cfg_min_batch, 80), (cfg_min_batch, 80, 4)]
        opt_shape = [(cfg_opt_batch, 80, 1000), (cfg_opt_batch, 1, 1000), (cfg_opt_batch, 80, 1000), (cfg_opt_batch,), (cfg_opt_batch, 80), (cfg_opt_batch, 80, 1000)]
        # FP32 requires more memory per element, so reduce max_shape to avoid OOM during TRT build
        # FP32: max 3000 frames (~37s audio), FP16: max 6000 frames (~75s audio)
        max_frames = 6000 if fp16 else 3000
        max_shape = [(cfg_max_batch, 80, max_frames), (cfg_max_batch, 1, max_frames), (cfg_max_batch, 80, max_frames), (cfg_max_batch,), (cfg_max_batch, 80), (cfg_max_batch, 80, max_frames)]
        input_names = ["x", "mask", "mu", "t", "spks", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}
    
    def load_trt(self, flow_decoder_estimator_model: str, flow_decoder_onnx_model: str, trt_concurrent: int = 1, fp16: bool = False, max_batch_size: int = 1):
        """Load TensorRT engine for Flow decoder (significant speedup)."""
        assert torch.cuda.is_available(), 'TensorRT only supports GPU!'

        # Use file lock to serialize ONNX export + TRT build across processes
        lock_path = flow_decoder_estimator_model + '.lock'
        with open(lock_path, 'w') as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # Auto-export ONNX if missing. Some model releases do not ship this file.
                if not os.path.exists(flow_decoder_onnx_model) or os.path.getsize(flow_decoder_onnx_model) == 0:
                    from cosyvoice.utils.file_utils import export_flow_decoder_estimator_onnx
                    export_flow_decoder_estimator_onnx(
                        estimator=self.flow.decoder.estimator,
                        onnx_path=flow_decoder_onnx_model,
                        device=self.device,
                    )

                if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
                    logging.info(f'Converting ONNX to TensorRT: {flow_decoder_onnx_model} -> {flow_decoder_estimator_model}')
                    convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(fp16, max_batch_size=max_batch_size), flow_decoder_onnx_model, fp16)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        
        del self.flow.decoder.estimator
        
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        
        assert estimator_engine is not None, f'Failed to load TensorRT engine: {flow_decoder_estimator_model}'
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)
        logging.info(f'TensorRT engine loaded: {flow_decoder_estimator_model}')

    # ── Cache flow TRT (CausalConditionalDecoder / UNet with KV cache) ─────

    def get_trt_kwargs_cache(self, fp16: bool = True, max_batch_size: int = 1):
        """TRT optimization profiles for cache-based UNet decoder (CausalConditionalDecoder).

        Dynamic axes: batch (2*B for CFG), seq_len (mel frames), cache_in_len (KV cache length).
        Requires ONNX exported with dynamic batch (dim 0 = 'batch').
        """
        cfg_min_batch = 2
        cfg_max_batch = max(2, int(max_batch_size) * 2)
        cfg_opt_batch = cfg_max_batch
        min_seq, opt_seq, max_seq = 4, 50, 1000
        min_cache, opt_cache, max_cache = 0, 50, 50
        input_names = [
            "x", "mask", "mu", "t", "spks", "cond",
            "down_blocks_conv_cache", "down_blocks_kv_cache",
            "mid_blocks_conv_cache", "mid_blocks_kv_cache",
            "up_blocks_conv_cache", "up_blocks_kv_cache",
            "final_blocks_conv_cache",
        ]
        min_shape = [
            # x, mask, mu
            (cfg_min_batch, 80, min_seq), (cfg_min_batch, 1, min_seq), (cfg_min_batch, 80, min_seq),
            # t, spks, cond
            (cfg_min_batch,), (cfg_min_batch, 80), (cfg_min_batch, 80, min_seq),
            # conv caches: down(1,B,832,2), mid(12,B,512,2), up(1,B,1024,2), final(B,256,2)
            (1, cfg_min_batch, 832, 2), (1, 4, cfg_min_batch, min_cache, 512, 2),
            (12, cfg_min_batch, 512, 2), (12, 4, cfg_min_batch, min_cache, 512, 2),
            (1, cfg_min_batch, 1024, 2), (1, 4, cfg_min_batch, min_cache, 512, 2),
            (cfg_min_batch, 256, 2),
        ]
        opt_shape = [
            (cfg_opt_batch, 80, opt_seq), (cfg_opt_batch, 1, opt_seq), (cfg_opt_batch, 80, opt_seq),
            (cfg_opt_batch,), (cfg_opt_batch, 80), (cfg_opt_batch, 80, opt_seq),
            (1, cfg_opt_batch, 832, 2), (1, 4, cfg_opt_batch, opt_cache, 512, 2),
            (12, cfg_opt_batch, 512, 2), (12, 4, cfg_opt_batch, opt_cache, 512, 2),
            (1, cfg_opt_batch, 1024, 2), (1, 4, cfg_opt_batch, opt_cache, 512, 2),
            (cfg_opt_batch, 256, 2),
        ]
        max_shape = [
            (cfg_max_batch, 80, max_seq), (cfg_max_batch, 1, max_seq), (cfg_max_batch, 80, max_seq),
            (cfg_max_batch,), (cfg_max_batch, 80), (cfg_max_batch, 80, max_seq),
            (1, cfg_max_batch, 832, 2), (1, 4, cfg_max_batch, max_cache, 512, 2),
            (12, cfg_max_batch, 512, 2), (12, 4, cfg_max_batch, max_cache, 512, 2),
            (1, cfg_max_batch, 1024, 2), (1, 4, cfg_max_batch, max_cache, 512, 2),
            (cfg_max_batch, 256, 2),
        ]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def load_trt_cache(self, flow_decoder_estimator_model: str, flow_decoder_onnx_model: str,
                       trt_concurrent: int = 1, fp16: bool = False, max_batch_size: int = 1):
        """Load TRT engine for cache-based CausalConditionalDecoder (UNet with KV cache).

        Auto-exports ONNX (with dynamic batch) if missing, then builds TRT plan.
        """
        assert torch.cuda.is_available(), 'TensorRT only supports GPU!'

        # Use file lock to serialize ONNX export + TRT build across processes
        lock_path = flow_decoder_estimator_model + '.lock'
        with open(lock_path, 'w') as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # Auto-export ONNX if missing
                if not os.path.exists(flow_decoder_onnx_model) or os.path.getsize(flow_decoder_onnx_model) == 0:
                    from cosyvoice.utils.file_utils import export_cache_flow_decoder_onnx
                    export_cache_flow_decoder_onnx(
                        estimator=self.flow.decoder.estimator,
                        onnx_path=flow_decoder_onnx_model,
                        device=self.device,
                        flow_decoder_required_cache_size=self.flow_decoder_required_cache_size,
                        flow_n_timesteps=self.flow_n_timesteps,
                    )

                # Build TRT plan if missing
                if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
                    logging.info(f'Converting cache flow ONNX to TRT: {flow_decoder_onnx_model} -> {flow_decoder_estimator_model}')
                    convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs_cache(fp16, max_batch_size), flow_decoder_onnx_model, fp16)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

        del self.flow.decoder.estimator

        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())

        assert estimator_engine is not None, f'Failed to load TRT engine: {flow_decoder_estimator_model}'
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)
        logging.info(f'Cache flow TRT engine loaded (max_batch={max_batch_size}): {flow_decoder_estimator_model}')

    def _expand_batch_tensor(self, tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if tensor.dim() == 0:
            raise ValueError(f'{name} must have a batch dimension')
        if tensor.size(0) == batch_size:
            return tensor
        if tensor.size(0) == 0:
            return tensor
        if tensor.size(0) == 1:
            expand_sizes = [batch_size] + [-1] * (tensor.dim() - 1)
            return tensor.expand(*expand_sizes)
        raise ValueError(f'{name} batch size mismatch: expected 1 or {batch_size}, got {tensor.size(0)}')

    def _resolve_lengths(
        self,
        lengths: Optional[torch.Tensor],
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        batch_size = tensor.size(0)
        if lengths is None:
            if tensor.dim() < 2:
                raise ValueError(f'Cannot infer {name} lengths from tensor with shape {tuple(tensor.shape)}')
            lengths = torch.full((batch_size,), tensor.size(1), dtype=torch.int32)
        if lengths.dim() == 0:
            lengths = lengths.reshape(1)
        lengths = lengths.to(dtype=torch.int32)
        if lengths.size(0) == batch_size:
            return lengths
        if lengths.size(0) == 1:
            return lengths.expand(batch_size)
        raise ValueError(f'{name} length batch mismatch: expected 1 or {batch_size}, got {lengths.size(0)}')

    def _queue_audio_chunk(self, audio_queue: queue.Queue, audio: torch.Tensor, sample_idx: int, finalize: bool = False):
        if audio.numel() == 0:
            return
        if audio.is_cuda:
            audio_cpu = torch.empty_like(audio, device='cpu', pin_memory=True)
            audio_cpu.copy_(audio, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
            audio_queue.put({'tts_speech': audio_cpu, 'sample_idx': sample_idx, 'finalize': finalize, '_ready_event': ready})
        else:
            audio_queue.put({'tts_speech': audio, 'sample_idx': sample_idx, 'finalize': finalize})

    def _index_batch_tensor(self, tensor: torch.Tensor, sample_indices: List[int]) -> torch.Tensor:
        if len(sample_indices) == 0:
            raise ValueError('sample_indices must not be empty')
        index = torch.tensor(sample_indices, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, index)

    def _expand_shared_batch_tensor(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.size(0) == batch_size:
            return tensor
        if tensor.size(0) != 1:
            raise ValueError(f'Expected shared tensor batch size 1 or {batch_size}, got {tensor.size(0)}')
        expand_sizes = [batch_size] + [-1] * (tensor.dim() - 1)
        return tensor.expand(*expand_sizes)

    def _run_hift_group(
        self,
        sample_indices: List[int],
        sample_mels: List[torch.Tensor],
        finalize: bool,
        speech_offset: List[int],
        audio_queue: queue.Queue,
        mel_lengths: Optional[List[int]] = None,
    ):
        if len(sample_indices) == 0:
            return 0.0, 0

        total_hift_elapsed = 0.0
        hift_call_count = 0

        buckets: Dict[int, List[int]] = {}
        for local_idx, sample_mel in enumerate(sample_mels):
            mel_len = int(sample_mel.shape[2])
            buckets.setdefault(mel_len, []).append(local_idx)

        for local_indices in buckets.values():
            bucket_sample_indices = [sample_indices[i] for i in local_indices]
            batch_mel = torch.cat([sample_mels[i] for i in local_indices], dim=0)

            hift_start = time.time()
            tts_speech, _ = self.hift.inference(
                speech_feat=batch_mel.float(),
                finalize=finalize
            )
            total_hift_elapsed += time.time() - hift_start
            hift_call_count += 1

            for batch_idx, sample_idx in enumerate(bucket_sample_indices):
                local_idx = local_indices[batch_idx]
                audio = tts_speech[batch_idx: batch_idx + 1, speech_offset[sample_idx]:]
                # trim padding-induced tail noise on finalize with batched input
                if finalize and mel_lengths is not None and len(bucket_sample_indices) > 1:
                    expected_audio_len = mel_lengths[local_idx] * 480
                    trim_len = expected_audio_len - speech_offset[sample_idx]
                    if trim_len > 0 and trim_len < audio.shape[1]:
                        audio = audio[:, :trim_len]
                speech_offset[sample_idx] += audio.shape[1]
                self._queue_audio_chunk(audio_queue, audio, sample_idx, finalize=finalize)

        return total_hift_elapsed, hift_call_count

    def _process_flow_group(
        self,
        sample_indices: List[int],
        finalize: bool,
        tokens_snapshot: List[List[int]],
        token_offset: List[int],
        mel_cache: List[Optional[torch.Tensor]],
        speech_offset: List[int],
        audio_queue: queue.Queue,
        flow_prompt_token_gpu: torch.Tensor,
        flow_prompt_token_len_gpu: torch.Tensor,
        prompt_feat_gpu: torch.Tensor,
        prompt_feat_len_gpu: torch.Tensor,
        flow_embedding_gpu: torch.Tensor,
        prompt_token_pad: torch.Tensor,
        flow_caches: Optional[List[dict]] = None,
        prompt_cleared: Optional[List[bool]] = None,
    ):
        if len(sample_indices) == 0:
            return 0.0, 0.0, 0, 0, False

        # ────────────────────────────────────────────────────────────────────
        # Cache flow path (CausalMaskedDiffWithXvec)
        # Batched flow.inference with merged encoder/decoder cache
        # Only sends NEW chunk tokens (not full accumulated sequence)
        # ────────────────────────────────────────────────────────────────────
        if self.use_flow_cache and flow_caches is not None:
            dtype = prompt_feat_gpu.dtype
            this_hop_len = self.token_hop_len

            # ── Collect chunk tokens per sample ──────────────────────────
            batch_chunk_tokens = []
            batch_chunk_lens = []
            valid_indices = []        # sample_indices with actual tokens
            valid_positions = []      # position in sample_indices list

            for pos, sample_idx in enumerate(sample_indices):
                sample_tokens = tokens_snapshot[sample_idx]
                if finalize:
                    chunk_tokens = sample_tokens[token_offset[sample_idx]:]
                else:
                    chunk_end = token_offset[sample_idx] + this_hop_len + self.flow.pre_lookahead_len
                    chunk_tokens = sample_tokens[token_offset[sample_idx]:chunk_end]

                if not chunk_tokens:
                    continue
                batch_chunk_tokens.append(
                    torch.tensor(chunk_tokens, dtype=torch.int32, device=self.device)
                )
                batch_chunk_lens.append(len(chunk_tokens))
                valid_indices.append(sample_idx)
                valid_positions.append(pos)

            # Handle case where no samples have tokens
            if not valid_indices:
                sample_mels = [torch.zeros(1, 80, 0, device=self.device, dtype=dtype)
                               for _ in sample_indices]
                hift_elapsed, hift_calls = self._run_hift_group(
                    sample_indices=sample_indices, sample_mels=sample_mels,
                    finalize=finalize, speech_offset=speech_offset,
                    audio_queue=audio_queue,
                )
                return 0.0, hift_elapsed, 0, hift_calls, False

            B = len(valid_indices)

            # ── Batch token tensors (pad if needed) ──────────────────────
            tok_batch = torch.nn.utils.rnn.pad_sequence(
                batch_chunk_tokens, batch_first=True, padding_value=0
            )  # [B, max_chunk_len]
            tok_len_batch = torch.tensor(
                batch_chunk_lens, dtype=torch.int32, device=self.device
            )  # [B]

            # ── Batch prompts ────────────────────────────────────────────
            pt_list, pt_len_list, pf_list, pf_len_list = [], [], [], []
            for sample_idx in valid_indices:
                if not prompt_cleared[sample_idx]:
                    s_pt_len = int(flow_prompt_token_len_gpu[sample_idx].item())
                    s_pf_len = int(prompt_feat_len_gpu[sample_idx].item())
                    pt_list.append(flow_prompt_token_gpu[sample_idx, :s_pt_len])
                    pt_len_list.append(s_pt_len)
                    pf_list.append(prompt_feat_gpu[sample_idx, :s_pf_len])
                    pf_len_list.append(s_pf_len)
                else:
                    pt_list.append(torch.zeros(0, dtype=torch.int32, device=self.device))
                    pt_len_list.append(0)
                    pf_list.append(torch.zeros(0, 80, device=self.device, dtype=dtype))
                    pf_len_list.append(0)

            pt_batch = torch.nn.utils.rnn.pad_sequence(
                pt_list, batch_first=True, padding_value=0
            )  # [B, max_pt_len]
            pt_len_batch = torch.tensor(
                pt_len_list, dtype=torch.int32, device=self.device
            )
            pf_batch = torch.nn.utils.rnn.pad_sequence(
                pf_list, batch_first=True, padding_value=0.0
            )  # [B, max_pf_len, 80]
            pf_len_batch = torch.tensor(
                pf_len_list, dtype=torch.int32, device=self.device
            )
            emb_batch = torch.cat(
                [flow_embedding_gpu[si:si+1] for si in valid_indices], dim=0
            )  # [B, 192]

            # ── Merge per-sample caches → single batched cache ───────────
            merged_cache = self._merge_flow_caches(flow_caches, valid_indices)

            # ── Single batched flow.inference call ───────────────────────
            flow_start = time.time()
            tts_mel_batch, merged_cache, flow_mel_len2_list = self.flow.inference(
                token=tok_batch,
                token_len=tok_len_batch,
                prompt_token=pt_batch,
                prompt_token_len=pt_len_batch,
                prompt_feat=pf_batch,
                prompt_feat_len=pf_len_batch,
                embedding=emb_batch,
                cache=merged_cache,
                finalize=finalize,
                flow_step=self.flow_n_timesteps,
            )
            total_flow_time = time.time() - flow_start

            # ── Split cache back to per-sample ───────────────────────────
            self._split_flow_caches(merged_cache, flow_caches, valid_indices)

            # ── Distribute results to per-sample mels ────────────────────
            # Build full sample_mels list (including empties for skipped samples)
            sample_mels = []
            actual_mel_lengths = []
            produced_any = False
            valid_set = set(valid_indices)
            valid_iter = iter(range(B))

            for pos, sample_idx in enumerate(sample_indices):
                if sample_idx not in valid_set:
                    sample_mels.append(torch.zeros(1, 80, 0, device=self.device, dtype=dtype))
                    actual_mel_lengths.append(0)
                    continue

                vi = next(valid_iter)
                mel_new = tts_mel_batch[vi:vi+1].float()

                if not prompt_cleared[sample_idx]:
                    prompt_cleared[sample_idx] = True

                cache_mel_len = int(mel_cache[sample_idx].shape[2]) if mel_cache[sample_idx] is not None else 0
                actual_total_mel_len = cache_mel_len + flow_mel_len2_list[vi]

                mel = mel_new if mel_cache[sample_idx] is None else torch.cat([mel_cache[sample_idx], mel_new], dim=2)
                mel_cache[sample_idx] = None if finalize else mel
                sample_mels.append(mel)
                actual_mel_lengths.append(actual_total_mel_len)
                produced_any = produced_any or mel_new.shape[2] > 0

                # Advance offset
                sample_tokens = tokens_snapshot[sample_idx]
                token_offset[sample_idx] = (
                    len(sample_tokens) if finalize
                    else token_offset[sample_idx] + this_hop_len
                )

            hift_elapsed, hift_calls = self._run_hift_group(
                sample_indices=sample_indices,
                sample_mels=sample_mels,
                finalize=finalize,
                speech_offset=speech_offset,
                audio_queue=audio_queue,
                mel_lengths=actual_mel_lengths if finalize else None,
            )
            return total_flow_time, hift_elapsed, 1, hift_calls, produced_any

        # ────────────────────────────────────────────────────────────────────
        # Original DiT path (stateless, no cache)
        # Sends ALL tokens from start to current position each call
        # ────────────────────────────────────────────────────────────────────
        batch_tokens = []
        batch_token_lens = []
        valid_token_ends = []
        hop_lens = {}

        for sample_idx in sample_indices:
            sample_tokens = tokens_snapshot[sample_idx]
            this_prompt_pad = int(prompt_token_pad[sample_idx].item())
            this_hop_len = self.token_hop_len + this_prompt_pad if token_offset[sample_idx] == 0 else self.token_hop_len
            hop_lens[sample_idx] = this_hop_len

            token_end = len(sample_tokens) if finalize else token_offset[sample_idx] + this_hop_len + self.flow.pre_lookahead_len
            batch_tokens.append(torch.tensor(sample_tokens[:token_end], dtype=torch.int32, device=self.device))
            batch_token_lens.append(token_end)
            valid_token_ends.append(len(sample_tokens) if finalize else token_offset[sample_idx] + this_hop_len)

        batch_tokens = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True, padding_value=0)
        token_len_gpu = torch.tensor(batch_token_lens, dtype=torch.int32, device=self.device)

        batch_size = len(sample_indices)
        flow_start = time.time()
        # with torch.amp.autocast('cuda', enabled=self.fp16):
        tts_mel, _ = self.flow.inference(
            token=batch_tokens,
            token_len=token_len_gpu,
            prompt_token=self._expand_shared_batch_tensor(flow_prompt_token_gpu[:1], batch_size),
            prompt_token_len=self._expand_shared_batch_tensor(flow_prompt_token_len_gpu[:1], batch_size),
            prompt_feat=self._expand_shared_batch_tensor(prompt_feat_gpu[:1], batch_size),
            prompt_feat_len=self._expand_shared_batch_tensor(prompt_feat_len_gpu[:1], batch_size),
            embedding=self._index_batch_tensor(flow_embedding_gpu, sample_indices),
            streaming=True,
            finalize=finalize
        )
        flow_elapsed = time.time() - flow_start

        ratio = self.flow.token_mel_ratio
        sample_mels = []
        produced_any = False
        for batch_idx, sample_idx in enumerate(sample_indices):
            old_offset = token_offset[sample_idx]
            valid_token_end = valid_token_ends[batch_idx]
            valid_mel_len = valid_token_end * ratio

            sample_mel = tts_mel[batch_idx: batch_idx + 1, :, :valid_mel_len]
            mel_new = sample_mel[:, :, old_offset * ratio:]
            mel = mel_new if mel_cache[sample_idx] is None else torch.cat([mel_cache[sample_idx], mel_new], dim=2)
            mel_cache[sample_idx] = None if finalize else mel
            sample_mels.append(mel)
            token_offset[sample_idx] = len(tokens_snapshot[sample_idx]) if finalize else old_offset + hop_lens[sample_idx]
            produced_any = produced_any or mel_new.shape[2] > 0

        hift_mel_lengths = [int(m.shape[2]) for m in sample_mels] if finalize else None
        hift_elapsed, hift_calls = self._run_hift_group(
            sample_indices=sample_indices,
            sample_mels=sample_mels,
            finalize=finalize,
            speech_offset=speech_offset,
            audio_queue=audio_queue,
            mel_lengths=hift_mel_lengths,
        )

        return flow_elapsed, hift_elapsed, 1, hift_calls, produced_any
    
    def _llm_job(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        llm_prompt_speech_token: torch.Tensor,
        llm_prompt_speech_token_len: torch.Tensor,
        llm_embedding: torch.Tensor,
        LM_latents: torch.Tensor,
        tokens_list: list,
        llm_end_flag: dict,
        tokens_lock: threading.Lock,
        llm_sample_done: Optional[list] = None,
    ):
        """
        LLM token generation - runs in dedicated thread with its own CUDA stream.
        Never blocked by Flow/Hift operations.
        """
        llm_start_time = time.time()
        token_count = 0
        
        # Pre-move tensors to device once
        text_gpu = text.to(self.device)
        text_len_gpu = text_len.to(self.device, dtype=torch.int32)
        prompt_text_gpu = prompt_text.to(self.device)
        prompt_text_len_gpu = prompt_text_len.to(self.device, dtype=torch.int32)
        prompt_speech_token_gpu = llm_prompt_speech_token.to(self.device)
        prompt_speech_token_len_gpu = llm_prompt_speech_token_len.to(self.device, dtype=torch.int32)
        embedding_gpu = llm_embedding.to(self.device)
        LM_latents_gpu = LM_latents.to(self.device)
        try:
            llm_context = torch.cuda.stream(self.llm_stream) if self.llm_stream else nullcontext()
            
            with llm_context, torch.inference_mode(), torch.amp.autocast('cuda',enabled=self.fp16):
            # with llm_context, torch.inference_mode():
                for token in self.llm.inference(
                    text=text_gpu,
                    text_len=text_len_gpu,
                    prompt_text=prompt_text_gpu,
                    prompt_text_len=prompt_text_len_gpu,
                    prompt_speech_token=prompt_speech_token_gpu,
                    prompt_speech_token_len=prompt_speech_token_len_gpu,
                    embedding=embedding_gpu,
                    LM_latents=LM_latents_gpu
                ):
                    if torch.is_tensor(token):
                        step_tokens = token.detach().cpu().reshape(-1).tolist()
                    else:
                        step_tokens = [int(token)]
                    with tokens_lock:
                        for sample_idx, step_token in enumerate(step_tokens):
                            if step_token is None or int(step_token) < 0:
                                if llm_sample_done is not None and sample_idx < len(llm_sample_done):
                                    llm_sample_done[sample_idx] = True
                                continue
                            tokens_list[sample_idx].append(int(step_token))
                            token_count += 1

            llm_duration = time.time() - llm_start_time
            tokens_per_sec = token_count / llm_duration if llm_duration > 0 else 0
            logging.info(
                f'[LLM] duration={llm_duration:.3f}s, tokens={token_count}, tokens/s={tokens_per_sec:.2f} (wall-clock, async)'
            )
            
        except Exception as e:
            logging.error(f'[LLM] Error: {e}', exc_info=True)
        finally:
            # Cleanup GPU tensors
            del text_gpu, text_len_gpu, prompt_text_gpu, prompt_text_len_gpu
            del prompt_speech_token_gpu, prompt_speech_token_len_gpu, embedding_gpu
            llm_end_flag['done'] = True
    
    def _flow_hift_job(
        self,
        tokens_list: list,
        tokens_lock: threading.Lock,
        llm_end_flag: dict,
        audio_queue: queue.Queue,
        flow_prompt_token_gpu: torch.Tensor,
        flow_prompt_token_len_gpu: torch.Tensor,
        prompt_feat_gpu: torch.Tensor,
        prompt_feat_len_gpu: torch.Tensor,
        flow_embedding_gpu: torch.Tensor,
        prompt_token_pad: torch.Tensor,
        llm_sample_done: Optional[list] = None,
    ):
        """
        Flow + Hift processing - runs in dedicated thread.
        Blocking operations here don't affect LLM at all.
        """
        batch_size = len(tokens_list)
        token_offset = [0 for _ in range(batch_size)]
        mel_cache: List[Optional[torch.Tensor]] = [None for _ in range(batch_size)]
        speech_offset = [0 for _ in range(batch_size)]
        sample_finalized = [False] * batch_size

        # Cache flow state (CausalMaskedDiffWithXvec)
        if self.use_flow_cache:
            flow_caches = [self._init_flow_cache(batch_size=1) for _ in range(batch_size)]
            prompt_cleared = [False] * batch_size
        else:
            flow_caches = None
            prompt_cleared = None

        # Timing accumulators
        total_flow_time = 0.0
        total_hift_time = 0.0
        flow_call_count = 0
        hift_call_count = 0

        try:
            with torch.inference_mode():
                while True:
                    # Minimal polling
                    time.sleep(0.0005)

                    with tokens_lock:
                        tokens_snapshot = [list(sample_tokens) for sample_tokens in tokens_list]
                        sample_done_snapshot = (
                            list(llm_sample_done)
                            if llm_sample_done is not None
                            else [llm_end_flag['done']] * batch_size
                        )
                    stream_ready: List[int] = []
                    final_ready: List[int] = []
                    for sample_idx, sample_tokens in enumerate(tokens_snapshot):
                        if sample_finalized[sample_idx]:
                            continue
                        this_prompt_pad = int(prompt_token_pad[sample_idx].item())
                        # Cache flow: no prompt_pad (prompt handled separately by flow.inference)
                        if self.use_flow_cache:
                            this_hop_len = self.token_hop_len
                        else:
                            this_hop_len = self.token_hop_len + this_prompt_pad if token_offset[sample_idx] == 0 else self.token_hop_len
                        required_tokens = this_hop_len + self.flow.pre_lookahead_len
                        available = len(sample_tokens) - token_offset[sample_idx]

                        if available >= required_tokens:
                            stream_ready.append(sample_idx)
                        elif sample_done_snapshot[sample_idx]:
                            if available > 0:
                                final_ready.append(sample_idx)
                            else:
                                # LLM done but no tokens to process - mark finalized
                                sample_finalized[sample_idx] = True

                    processed_any = False
                    if stream_ready:
                        flow_elapsed, hift_elapsed, flow_calls, hift_calls, did_process = self._process_flow_group(
                            sample_indices=stream_ready,
                            finalize=False,
                            tokens_snapshot=tokens_snapshot,
                            token_offset=token_offset,
                            mel_cache=mel_cache,
                            speech_offset=speech_offset,
                            audio_queue=audio_queue,
                            flow_prompt_token_gpu=flow_prompt_token_gpu,
                            flow_prompt_token_len_gpu=flow_prompt_token_len_gpu,
                            prompt_feat_gpu=prompt_feat_gpu,
                            prompt_feat_len_gpu=prompt_feat_len_gpu,
                            flow_embedding_gpu=flow_embedding_gpu,
                            prompt_token_pad=prompt_token_pad,
                            flow_caches=flow_caches,
                            prompt_cleared=prompt_cleared,
                        )
                        total_flow_time += flow_elapsed
                        total_hift_time += hift_elapsed
                        flow_call_count += flow_calls
                        hift_call_count += hift_calls
                        processed_any = processed_any or did_process

                    if final_ready:
                        flow_elapsed, hift_elapsed, flow_calls, hift_calls, did_process = self._process_flow_group(
                            sample_indices=final_ready,
                            finalize=True,
                            tokens_snapshot=tokens_snapshot,
                            token_offset=token_offset,
                            mel_cache=mel_cache,
                            speech_offset=speech_offset,
                            audio_queue=audio_queue,
                            flow_prompt_token_gpu=flow_prompt_token_gpu,
                            flow_prompt_token_len_gpu=flow_prompt_token_len_gpu,
                            prompt_feat_gpu=prompt_feat_gpu,
                            prompt_feat_len_gpu=prompt_feat_len_gpu,
                            flow_embedding_gpu=flow_embedding_gpu,
                            prompt_token_pad=prompt_token_pad,
                            flow_caches=flow_caches,
                            prompt_cleared=prompt_cleared,
                        )
                        total_flow_time += flow_elapsed
                        total_hift_time += hift_elapsed
                        flow_call_count += flow_calls
                        hift_call_count += hift_calls
                        processed_any = processed_any or did_process
                        for sample_idx in final_ready:
                            sample_finalized[sample_idx] = True

                    if all(sample_finalized) and not processed_any:
                        break
        
        except Exception as e:
            logging.error(f'[Flow+Hift] Error: {e}', exc_info=True)
        finally:
            # Log timing statistics
            logging.info(
                f'[Flow] total_time={total_flow_time:.3f}s, calls={flow_call_count}, '
                f'avg_per_call={total_flow_time/flow_call_count if flow_call_count > 0 else 0:.3f}s'
            )
            logging.info(
                f'[HiFT] total_time={total_hift_time:.3f}s, calls={hift_call_count}, '
                f'avg_per_call={total_hift_time/hift_call_count if hift_call_count > 0 else 0:.3f}s'
            )
            # Signal completion
            audio_queue.put(None)
    
    def tts_stream(
        self,
        text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_embedding: torch.Tensor = torch.zeros(0, 192),
        llm_embedding: torch.Tensor = torch.zeros(0, 192),
        prompt_text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        LM_latents: torch.Tensor = torch.zeros(1, 0, 2048),
        **kwargs
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        True pipeline streaming TTS.
        
        Architecture:
        - LLM Thread: generates tokens (never blocked by Flow/Hift)
        - Flow+Hift Thread: processes tokens into audio
        - Main Thread: yields audio chunks from queue
        
        This achieves ~80-90% of isolated LLM throughput.
        """
        batch_size = text.size(0)
        text_len = self._resolve_lengths(kwargs.get('text_len'), text, 'text')
        prompt_text = self._expand_batch_tensor(prompt_text, batch_size, 'prompt_text')
        prompt_text_len = self._resolve_lengths(kwargs.get('prompt_text_len'), prompt_text, 'prompt_text')
        llm_prompt_speech_token = self._expand_batch_tensor(llm_prompt_speech_token, batch_size, 'llm_prompt_speech_token')
        llm_prompt_speech_token_len = self._resolve_lengths(kwargs.get('llm_prompt_speech_token_len'), llm_prompt_speech_token, 'llm_prompt_speech_token')
        flow_prompt_speech_token = self._expand_batch_tensor(flow_prompt_speech_token, batch_size, 'flow_prompt_speech_token')
        flow_prompt_speech_token_len = self._resolve_lengths(kwargs.get('flow_prompt_speech_token_len'), flow_prompt_speech_token, 'flow_prompt_speech_token')
        prompt_speech_feat = self._expand_batch_tensor(prompt_speech_feat, batch_size, 'prompt_speech_feat')
        prompt_speech_feat_len = self._resolve_lengths(kwargs.get('prompt_speech_feat_len'), prompt_speech_feat, 'prompt_speech_feat')
        flow_embedding = self._expand_batch_tensor(flow_embedding, batch_size, 'flow_embedding')
        if llm_embedding.numel() != 0:
            llm_embedding = self._expand_batch_tensor(llm_embedding, batch_size, 'llm_embedding')
        if LM_latents.numel() != 0:
            LM_latents = self._expand_batch_tensor(LM_latents, batch_size, 'LM_latents')

        # Shared state
        tokens: list = [[] for _ in range(batch_size)]
        tokens_lock = threading.Lock()
        llm_end_flag = {'done': False}
        llm_sample_done: list = [False] * batch_size
        audio_queue: queue.Queue = queue.Queue(maxsize=4)  # Small buffer to limit memory
        
        # Pre-compute padding
        prompt_token_pad = (
            ((flow_prompt_speech_token_len + self.token_hop_len - 1) // self.token_hop_len) * self.token_hop_len
            - flow_prompt_speech_token_len
        ).to(dtype=torch.int32)
        
        # Pre-move tensors to device
        dtype = torch.float16 if self.fp16 else torch.float32
        flow_prompt_token_gpu = flow_prompt_speech_token.to(self.device)
        flow_prompt_token_len_gpu = flow_prompt_speech_token_len.to(self.device, dtype=torch.int32)
        prompt_feat_gpu = prompt_speech_feat.to(self.device, dtype=dtype)
        prompt_feat_len_gpu = prompt_speech_feat_len.to(self.device, dtype=torch.int32)
        flow_embedding_gpu = flow_embedding.to(self.device, dtype=dtype)
        # Start LLM thread
        llm_thread = threading.Thread(
            target=self._llm_job,
            args=(text, text_len, prompt_text, prompt_text_len, llm_prompt_speech_token, llm_prompt_speech_token_len, llm_embedding, LM_latents,
                  tokens, llm_end_flag, tokens_lock, llm_sample_done),
            daemon=True
        )

        # Start Flow+Hift thread
        flow_hift_thread = threading.Thread(
            target=self._flow_hift_job,
            args=(tokens, tokens_lock, llm_end_flag, audio_queue,
                  flow_prompt_token_gpu, flow_prompt_token_len_gpu,
                  prompt_feat_gpu, prompt_feat_len_gpu, flow_embedding_gpu,
                  prompt_token_pad, llm_sample_done),
            daemon=True
        )
        
        llm_thread.start()
        flow_hift_thread.start()
        
        try:
            # Main thread just yields from queue
            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=30.0)
                    if audio_chunk is None:  # Sentinel - processing complete
                        break
                    ready = audio_chunk.pop('_ready_event', None)
                    if ready is not None:
                        ready.synchronize()
                    yield audio_chunk
                except queue.Empty:
                    logging.warning('[Main] Timeout waiting for audio chunk')
                    break
            
            # Wait for threads
            llm_thread.join(timeout=5.0)
            flow_hift_thread.join(timeout=5.0)
            
        finally:
            pass  # GPU tensors will be freed when function exits
    
    def tts_stream_external_llm(
        self,
        tokens_list: list,
        tokens_lock: 'threading.Lock',
        llm_end_flag: dict,
        flow_embedding: torch.Tensor = torch.zeros(0, 192),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        **kwargs
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        True pipeline streaming TTS with external LLM (e.g. TRT-LLM).
        
        Similar to tts_stream but LLM runs in an external thread.
        Flow+Hift runs in its own thread and processes tokens as they arrive.
        
        Args:
            tokens_list: Shared list where external LLM appends tokens
            tokens_lock: Lock for thread-safe access to tokens_list
            llm_end_flag: Dict with 'done' key set to True when LLM finishes
            flow_embedding: Speaker embedding for flow
            flow_prompt_speech_token: Prompt speech tokens for flow
            prompt_speech_feat: Prompt speech features
        
        Yields:
            Audio chunks as dictionaries with 'tts_speech' key
        """
        batch_size = flow_prompt_speech_token.size(0)
        flow_prompt_speech_token = self._expand_batch_tensor(flow_prompt_speech_token, batch_size, 'flow_prompt_speech_token')
        flow_prompt_speech_token_len = self._resolve_lengths(kwargs.get('flow_prompt_speech_token_len'), flow_prompt_speech_token, 'flow_prompt_speech_token')
        prompt_speech_feat = self._expand_batch_tensor(prompt_speech_feat, batch_size, 'prompt_speech_feat')
        prompt_speech_feat_len = self._resolve_lengths(kwargs.get('prompt_speech_feat_len'), prompt_speech_feat, 'prompt_speech_feat')
        flow_embedding = self._expand_batch_tensor(flow_embedding, batch_size, 'flow_embedding')
        if batch_size == 1 and (len(tokens_list) == 0 or not isinstance(tokens_list[0], list)):
            tokens_list = [tokens_list]

        # Audio queue for output
        audio_queue: queue.Queue = queue.Queue(maxsize=4)
        
        # Pre-compute padding
        prompt_token_pad = (
            ((flow_prompt_speech_token_len + self.token_hop_len - 1) // self.token_hop_len) * self.token_hop_len
            - flow_prompt_speech_token_len
        ).to(dtype=torch.int32)
        
        # Pre-move tensors to device
        dtype = torch.float16 if self.fp16 else torch.float32
        flow_prompt_token_gpu = flow_prompt_speech_token.to(self.device)
        flow_prompt_token_len_gpu = flow_prompt_speech_token_len.to(self.device, dtype=torch.int32)
        prompt_feat_gpu = prompt_speech_feat.to(self.device, dtype=dtype)
        prompt_feat_len_gpu = prompt_speech_feat_len.to(self.device, dtype=torch.int32)
        flow_embedding_gpu = flow_embedding.to(self.device, dtype=dtype)
        
        # Start Flow+Hift thread
        flow_hift_thread = threading.Thread(
            target=self._flow_hift_job,
            args=(tokens_list, tokens_lock, llm_end_flag, audio_queue,
                  flow_prompt_token_gpu, flow_prompt_token_len_gpu,
                  prompt_feat_gpu, prompt_feat_len_gpu, flow_embedding_gpu,
                  prompt_token_pad),
            daemon=True
        )
        flow_hift_thread.start()
        
        try:
            # Main thread yields audio from queue
            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=30.0)
                    if audio_chunk is None:  # Sentinel - processing complete
                        break
                    ready = audio_chunk.pop('_ready_event', None)
                    if ready is not None:
                        ready.synchronize()
                    yield audio_chunk
                except queue.Empty:
                    logging.warning('[Main] Timeout waiting for audio chunk')
                    break
            
            flow_hift_thread.join(timeout=5.0)
            
        finally:
            pass  # GPU tensors will be freed when function exits
    
    def tts(
        self,
        text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_embedding: torch.Tensor = torch.zeros(0, 192),
        llm_embedding: torch.Tensor = torch.zeros(0, 192),
        prompt_text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        LM_latents: torch.Tensor = torch.zeros(1, 0, 2048),
        speed: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Non-streaming TTS inference.
        
        Generates all speech tokens first, then converts them to audio in one pass.
        This is simpler and can be slightly faster for short texts, but has higher
        latency to first audio compared to streaming.
        
        Args:
            text: Input text tokens
            flow_embedding: Speaker embedding for flow
            llm_embedding: Speaker embedding for LLM
            prompt_text: Prompt text tokens
            llm_prompt_speech_token: Prompt speech tokens for LLM
            flow_prompt_speech_token: Prompt speech tokens for flow
            prompt_speech_feat: Prompt speech features
            speed: Speech speed multiplier (1.0 = normal)
        
        Returns:
            Dict with 'tts_speech' key containing full audio tensor [1, audio_len]
        """
        if text.size(0) != 1:
            raise NotImplementedError('Non-streaming tts() currently supports batch_size=1 only')

        # Shared state for LLM thread
        tokens: list = [[]]
        tokens_lock = threading.Lock()
        llm_end_flag = {'done': False}
        text_len = self._resolve_lengths(kwargs.get('text_len'), text, 'text')
        prompt_text_len = self._resolve_lengths(kwargs.get('prompt_text_len'), prompt_text, 'prompt_text')
        llm_prompt_speech_token_len = self._resolve_lengths(kwargs.get('llm_prompt_speech_token_len'), llm_prompt_speech_token, 'llm_prompt_speech_token')
        
        # Start LLM thread
        llm_thread = threading.Thread(
            target=self._llm_job,
            args=(text, text_len, prompt_text, prompt_text_len, llm_prompt_speech_token, llm_prompt_speech_token_len, llm_embedding, LM_latents,
                  tokens, llm_end_flag, tokens_lock),
            daemon=True
        )
        llm_thread.start()
        
        # Wait for LLM to finish generating all tokens
        llm_thread.join()
        
        # Get all tokens
        with tokens_lock:
            all_tokens = list(tokens[0])
        
        if not all_tokens:
            logging.warning('[TTS] No tokens generated')
            return {'tts_speech': torch.zeros(1, 0)}
        
        # Convert tokens to audio in one pass
        dtype = torch.float16 if self.fp16 else torch.float32
        tokens_gpu = torch.tensor(all_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
        token_len_gpu = torch.tensor([tokens_gpu.shape[1]], dtype=torch.int32, device=self.device)
        flow_prompt_token_gpu = flow_prompt_speech_token.to(self.device)
        flow_prompt_token_len_gpu = torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32, device=self.device)
        prompt_feat_gpu = prompt_speech_feat.to(self.device, dtype=dtype)
        prompt_feat_len_gpu = torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32, device=self.device)
        flow_embedding_gpu = flow_embedding.to(self.device, dtype=dtype)
        
        with torch.inference_mode():
            # Flow inference - non-streaming mode
            flow_start = time.time()
            if self.use_flow_cache:
                # XVec: use cache-based inference (single pass, finalize=True)
                flow_cache = self._init_flow_cache()
                tts_mel, _ = self.flow.inference(
                    token=tokens_gpu,
                    token_len=token_len_gpu,
                    prompt_token=flow_prompt_token_gpu,
                    prompt_token_len=flow_prompt_token_len_gpu,
                    prompt_feat=prompt_feat_gpu,
                    prompt_feat_len=prompt_feat_len_gpu,
                    embedding=flow_embedding_gpu,
                    cache=flow_cache,
                    finalize=True,
                    flow_step=self.flow_n_timesteps,
                )
            else:
                # DiT: stateless inference
                tts_mel, _ = self.flow.inference(
                    token=tokens_gpu,
                    token_len=token_len_gpu,
                    prompt_token=flow_prompt_token_gpu,
                    prompt_token_len=flow_prompt_token_len_gpu,
                    prompt_feat=prompt_feat_gpu,
                    prompt_feat_len=prompt_feat_len_gpu,
                    embedding=flow_embedding_gpu,
                    streaming=False,
                    finalize=True
                )
            flow_elapsed = time.time() - flow_start
            logging.info(f'[Flow] non-streaming inference: {flow_elapsed:.3f}s')

            # Apply speed change if requested
            if speed != 1.0:
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')

            # Hift inference - non-streaming mode
            hift_start = time.time()
            tts_speech, _ = self.hift.inference(
                speech_feat=tts_mel.float(),
                finalize=True
            )
            hift_elapsed = time.time() - hift_start
            logging.info(f'[HiFT] non-streaming inference: {hift_elapsed:.3f}s')

        return {'tts_speech': tts_speech.cpu()}

    def tts_with_external_tokens(
        self,
        tokens: list,
        flow_embedding: torch.Tensor = torch.zeros(0, 192),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        speed: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Non-streaming TTS with pre-generated tokens (for TRT-LLM).
        
        Args:
            tokens: Pre-generated speech tokens
            flow_embedding: Speaker embedding for flow
            flow_prompt_speech_token: Prompt speech tokens for flow
            prompt_speech_feat: Prompt speech features
            speed: Speech speed multiplier (1.0 = normal)
        
        Returns:
            Dict with 'tts_speech' key containing full audio tensor [1, audio_len]
        """
        if not tokens:
            logging.warning('[TTS] No tokens provided')
            return {'tts_speech': torch.zeros(1, 0)}
        
        # Convert tokens to audio in one pass
        dtype = torch.float16 if self.fp16 else torch.float32
        tokens_gpu = torch.tensor(tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
        token_len_gpu = torch.tensor([tokens_gpu.shape[1]], dtype=torch.int32, device=self.device)
        flow_prompt_token_gpu = flow_prompt_speech_token.to(self.device)
        flow_prompt_token_len_gpu = torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32, device=self.device)
        prompt_feat_gpu = prompt_speech_feat.to(self.device, dtype=dtype)
        prompt_feat_len_gpu = torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32, device=self.device)
        flow_embedding_gpu = flow_embedding.to(self.device, dtype=dtype)
        
        with torch.inference_mode():
            # Flow inference - non-streaming mode
            flow_start = time.time()
            if self.use_flow_cache:
                flow_cache = self._init_flow_cache()
                tts_mel, _ = self.flow.inference(
                    token=tokens_gpu,
                    token_len=token_len_gpu,
                    prompt_token=flow_prompt_token_gpu,
                    prompt_token_len=flow_prompt_token_len_gpu,
                    prompt_feat=prompt_feat_gpu,
                    prompt_feat_len=prompt_feat_len_gpu,
                    embedding=flow_embedding_gpu,
                    cache=flow_cache,
                    finalize=True,
                    flow_step=self.flow_n_timesteps,
                )
            else:
                tts_mel, _ = self.flow.inference(
                    token=tokens_gpu,
                    token_len=token_len_gpu,
                    prompt_token=flow_prompt_token_gpu,
                    prompt_token_len=flow_prompt_token_len_gpu,
                    prompt_feat=prompt_feat_gpu,
                    prompt_feat_len=prompt_feat_len_gpu,
                    embedding=flow_embedding_gpu,
                    streaming=False,
                    finalize=True
                )
            flow_elapsed = time.time() - flow_start
            logging.info(f'[Flow] non-streaming inference: {flow_elapsed:.3f}s')

            # Apply speed change if requested
            if speed != 1.0:
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')

            # Hift inference - non-streaming mode
            hift_start = time.time()
            tts_speech, _ = self.hift.inference(
                speech_feat=tts_mel.float(),
                finalize=True
            )
            hift_elapsed = time.time() - hift_start
            logging.info(f'[HiFT] non-streaming inference: {hift_elapsed:.3f}s')

        return {'tts_speech': tts_speech.cpu()}
