#!/usr/bin/env python3

import sys
import time
import os
import logging
sys.path.append('third_party/Matcha-TTS')

import torch

from fastcosyvoice import FastCosyVoice3
from apis.utils import Custom_cap_function_fast


# Optimization for torch.compile (if used)
torch.set_float32_matmul_precision('high')

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model directory
MODEL_DIR = '/home/longtou.2024/mount/longtou/saved/fast_cosyvoice/Fun-CosyVoice3-0.5B'

# Reference audio file (3-10 sec, clean recording)
REFERENCE_AUDIO = '/home/longtou.2024/mount/longtou/saved/fast_cosyvoice/oneyoung_ref/oneyoung.wav'

# NOTE(longtou):
SPEAKER_PATHS = [
    '/home/longtou.2024/mount/sample_audio/원영_가이드.wav',
    '/home/longtou.2024/mount/sample_audio/킬링보이스_아이유_edit_short2.wav',
    '/home/longtou.2024/mount/sample_audio/kast_emo/oneyoung.wav',
    '/home/longtou.2024/mount/sample_audio/kast_emo/byeongheon.wav',
]
TEXT_PROMPTS=[
    "어? 재본 적이 없는데? 둘다 너무너무 탐이나는데 어머.. 러블리 히히~, 장원영의 칠초 인터뷰 시작하겠습니다.",
    "안녕하세요~. 킬링보이스에서 저를 찾으신다고 들어서 오늘 이렇게 조금.",
    "장원영의 칠초 인터뷰 시작하겠습니다.",
    "나는 정말 동엽이랑 워낙 친한 친구 사이니까.",
]

# Instruction for the model
INSTRUCTION = "You are a helpful assistant."

# TensorRT settings
USE_TRT_FLOW = False       # TensorRT for Flow decoder (~2.5x speedup)
USE_TRT_LLM = False        # TensorRT-LLM for LLM (~3x speedup)
TRT_LLM_DTYPE = 'float16'  # bfloat16/float16/float32
# Max tokens in KV-cache. 8192 tokens ≈ 100MB for Qwen2-0.5B.
# Minimum needed: max_input_len + max_output_len = 512 + 2048 = 2560 tokens.
TRT_LLM_KV_CACHE_TOKENS = 8192

# Inference wrapper without autograd (reduces allocations and graph leak risk)
USE_INFERENCE_MODE = True

# Texts for synthesis
SYNTHESIS_TEXTS = [
    "안녕하세요 카카오 엔터테인먼트 크루 여러분~",
    "아니 왜 하필 나한테 돌진한 거냐고!",
    "짐승도 암살에 쓰나?",
]


def load_prompt_text(audio_path: str, instruction: str = INSTRUCTION) -> str:
    """
    Loads transcription from txt file and forms prompt_text.
    
    Format prompt_text: "{instruction}<|endofprompt|>{transcription}"
    """
    txt_path = audio_path.rsplit('.', 1)[0] + '.txt'
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        transcription = f.read().strip()
    
    return f"{instruction}<|endofprompt|>{transcription}"




def synthesize_streaming(
    cosyvoice: FastCosyVoice3,
    text: str,
    spk_id: int,
):
    chunk_count = 0

    caption={'age':30, 'gender':'FEMALE', 'spoken_style':'독백체', 'emotion_style':'', 'emotion':'기쁨', 'intensity':1}
    caption=Custom_cap_function_fast(caption)

    infer_ctx = torch.inference_mode() if USE_INFERENCE_MODE else torch.no_grad()
    with infer_ctx:
        for pcm_bytes in cosyvoice.inference_zero_shot_stream(
            tts_text=text,
            prompt_text=TEXT_PROMPTS[spk_id],
            prompt_wav=SPEAKER_PATHS[spk_id],
            zero_shot_spk_id='',
            caption=caption,
        ):
            chunk_count += 1

            yield pcm_bytes


def load_model():
    # Check for reference audio
    if not os.path.exists(SPEAKER_PATHS[0]):
        logger.error(f"Reference audio not found: {SPEAKER_PATHS[0]}", exc_info=True)
        return

    print(f"\n🎤 Reference audio: {SPEAKER_PATHS[0]}")

    print("\n🔧 Loading FastCosyVoice3...")

    load_start = time.time()

    cosyvoice = FastCosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_trt=USE_TRT_FLOW,       # TensorRT for Flow decoder (~2.5x speedup)
        load_trt_llm=USE_TRT_LLM,    # TensorRT-LLM for LLM (~3x speedup)
        trt_llm_dtype=TRT_LLM_DTYPE,
        trt_llm_kv_cache_tokens=TRT_LLM_KV_CACHE_TOKENS,
    )

    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f} sec")

    # dtype diagnostics
    llm_dtype = next(cosyvoice.model.llm.parameters()).dtype
    flow_dtype = next(cosyvoice.model.flow.parameters()).dtype
    hift_dtype = next(cosyvoice.model.hift.parameters()).dtype
    print(f"📊 LLM dtype: {llm_dtype}, Flow dtype: {flow_dtype}, HiFT dtype: {hift_dtype}")

    sample_rate = cosyvoice.sample_rate
    print(f"📊 Sample rate: {sample_rate} Hz")

    # warmup
    print("Model Warmup Start")
    kwargs = {
        "text": "안녕하세요, 카카오 크루 여러분?",
        "spk_id": 0,
    }
    for _ in synthesize_streaming(cosyvoice, **kwargs):
        pass
    print("Model Warmup End")

    return cosyvoice


if __name__ == '__main__':
    model = load_model()

    kwargs = {
        "text": "안녕하세요, 카카오 크루 여러분?",
        "spk_id": 1,
    }
    queue = []
    chunk_cnt = 0
    for chunk in synthesize_streaming(model, **kwargs):
        audio_bytes = chunk
        queue.append(audio_bytes)
        chunk_cnt += 1
        print(f"청크 #{chunk_cnt}: {len(audio_bytes)} bytes")
    print("Done")

