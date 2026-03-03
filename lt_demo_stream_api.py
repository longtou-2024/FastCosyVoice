#!/usr/bin/env python3
"""
FastCosyVoice3 TTS - Parallel pipeline streaming inference with metrics measurement

Uses FastCosyVoice3 with parallel pipeline and TensorRT acceleration:
- LLM: TensorRT-LLM (~3x speedup) or PyTorch with torch.compile
- Flow: TensorRT (~2.5x speedup)
- Hift: PyTorch (f0_predictor on CPU)

Metrics:
- TTFB (Time To First Byte): time until first audio chunk is received
- RTF (Real-Time Factor): synthesis_time / audio_duration (< 1.0 = faster than real-time)
- Final audio duration
- Total generation time
"""

import sys
import time
import os
import logging
import wave
from pathlib import Path

sys.path.append('third_party/Matcha-TTS')

import torch
from cosyvoice.cli.cosyvoice import CosyVoice3
from fastcosyvoice import FastCosyVoice3


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

# Output directory
#OUTPUT_DIR = 'output/demo'

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


def apply_torch_compile(cosyvoice: FastCosyVoice3) -> None:
    """
    Applies torch.compile to LLM model for inference acceleration.
    
    Compiles the internal Qwen2ForCausalLM.model (Qwen2Model),
    which is used in forward_one_step for auto-generation.
    """
    # Path to Qwen2Model: cosyvoice.model.llm.llm.model.model
    # llm - CosyVoice3LM
    # llm.llm - Qwen2Encoder  
    # llm.llm.model - Qwen2ForCausalLM
    # llm.llm.model.model - Qwen2Model (what is actually called in forward_one_step)
    
    qwen2_model = cosyvoice.model.llm.llm.model.model
    logger.info(f"Compiling Qwen2Model: {type(qwen2_model).__name__}")
    
    compiled_model = torch.compile(qwen2_model, mode="default")
    cosyvoice.model.llm.llm.model.model = compiled_model
    
    logger.info("torch.compile applied to LLM")


def warmup_model(
    cosyvoice: FastCosyVoice3,
    prompt_text: str,
    spk_id: str,
) -> None:
    """
    Warms up the model by generating tokens to compile all execution paths.
    
    torch.compile creates different kernels for different input sizes,
    so the model needs to be warmed up on texts of different lengths.
    
    Args:
        cosyvoice: Initialized FastCosyVoice3 model
        prompt_text: Prompt text for generation
        spk_id: Speaker ID (should already be added via add_zero_shot_spk)
    """
    # Texts of different lengths to cover different input sizes
    warmup_texts = [
        # Short text (~50-100 LLM tokens)
        "Hello! How are you?",
        # Medium text (~100-200 LLM tokens)  
        "This is a test synthesis of medium-length text for model warmup.",
        # Long text (~200-400 LLM tokens)
        "This is a longer text for warmup. " * 3,
        # Very long text (~400+ LLM tokens)
        "Warming up the model on a long text for compilation. " * 5,
    ]
    
    warmup_start = time.time()
    
    # First pass - main compilation
    logger.info("Warmup: first pass (kernel compilation)...")
    for i, text in enumerate(warmup_texts):
        logger.info(f"  Warmup text {i+1}/{len(warmup_texts)}: {len(text)} characters")
        for _ in cosyvoice.inference_zero_shot_stream(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
        ):
            pass  # Just generate all chunks
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Second pass - ensure all paths are compiled
    logger.info("Warmup: second pass (stabilization)...")
    for text in warmup_texts:
        for _ in cosyvoice.inference_zero_shot_stream(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
        ):
            pass
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_time = time.time() - warmup_start
    logger.info(f"Warmup completed in {warmup_time:.2f} sec")


def synthesize_streaming(
    cosyvoice: FastCosyVoice3,
    text: str,
    prompt_text: str,
    spk_id: str,
    sample_rate: int,
):
    """
    Performs streaming synthesis of text through parallel pipeline and returns metrics.
    
    Args:
        cosyvoice: FastCosyVoice3 model
        text: Text for synthesis
        prompt_text: Reference audio transcription
        spk_id: Speaker ID
        sample_rate: Sample rate
    
    Returns:

    """
    #start_time = time.time()
    audio_chunks: list[bytes] = []
    chunk_count = 0

    infer_ctx = torch.inference_mode() if USE_INFERENCE_MODE else torch.no_grad()
    with infer_ctx:
        for pcm_bytes in cosyvoice.inference_zero_shot_stream(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
        ):
            chunk_count += 1

            yield pcm_bytes
            #audio_chunks.append(pcm_bytes)


def synthesize_streaming_basic(
    cosyvoice: CosyVoice3,
    text: str,
    prompt_text: str,
    spk_id: str,
    sample_rate: int,
):
    """
    Performs streaming synthesis of text through parallel pipeline and returns metrics.
    
    Args:
        cosyvoice: FastCosyVoice3 model
        text: Text for synthesis
        prompt_text: Reference audio transcription
        spk_id: Speaker ID
        sample_rate: Sample rate
    
    Returns:

    """

    # Create generator first (no computation yet)
    generator = cosyvoice.inference_zero_shot(
        tts_text=text,
        prompt_text=prompt_text,
        prompt_wav=REFERENCE_AUDIO,
        zero_shot_spk_id=spk_id,
        stream=True,
    )
    # Sync GPU before starting timer to ensure clean measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for model_output in generator:
        #chunk_count += 1
        
        # Get speech tensor
        speech = model_output['tts_speech']
        yield speech

def Custom_cap_function_fast(custon_json):
    # === Unpack input ===
    # age_val = int(custon_json['age'])
    genders = custon_json['gender']
    spoken_style = custon_json['spoken_style']
    emotion_style = custon_json['emotion_style']

    emotion = custon_json['emotion']
    intensity = custon_json['intensity']
    intensity_array=['','"약하게"','"중간정도의"','"강하게"']
    #speed = custon_json['pace']

    # if age
    if genders == "MALE":
        gender = '"남성"의 목소리로'
    elif genders == "FEMALE":
        gender = '"여성"의 목소리로'
    else: ### will be fix (단로로운 -> 단조로운)
        gender = ''
    if emotion_style:
        if spoken_style:
            if emotion=='무감정':
                emotion='"중립적인" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='기쁨':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+ ' "기쁜" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='슬픔':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "슬픈" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='분노':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "분노하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='당황':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "당황하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='불안':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "불안한" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='상처':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "상처받은" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
        else:
            if emotion=='무감정':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"중립적인" 감정'
            elif emotion=='기쁨':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"기쁜" 감정'
            elif emotion=='슬픔':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"슬픈" 감정'
            elif emotion=='분노':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"분노하는" 감정'
            elif emotion=='당황':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"당황하는" 감정'
            elif emotion=='불안':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"불안한" 감정'
            elif emotion=='상처':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"상처받은" 감정'
    else:
        if intensity_array:
            if emotion=='무감정':
                emotion='"중립적인" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='기쁨':
                emotion=intensity_array[intensity]+ ' "기쁜" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='슬픔':
                emotion=intensity_array[intensity]+' "슬픈" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='분노':
                emotion=intensity_array[intensity]+' "분노하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='당황':
                emotion=intensity_array[intensity]+' "당황하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='불안':
                emotion=intensity_array[intensity]+' "불안한" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='상처':
                emotion=intensity_array[intensity]+' "상처받은" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
        else:
            if emotion=='무감정':
                emotion='"중립적인" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='기쁨':
                emotion='"기쁜" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='슬픔':
                emotion='"슬픈" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='분노':
                emotion='"분노하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='당황':
                emotion='"당황하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='불안':
                emotion='"불안한" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='상처':
                emotion='"상처받은" 감정이고 '+'"'+spoken_style+'"'+' 스타일'

    parts = []

    if emotion: parts.append(emotion)

    caption=' '.join(parts)
    # === Final composition ===

    caption=emotion+'<|endofprompt|>'
    # === Final composition ===
    return caption

def synthesize_streaming_kayden(
    cosyvoice: FastCosyVoice3,
    text: str,
    prompt_text: str,
    spk_id: str,
    sample_rate: int,
):
    chunk_count = 0

    caption={'age':30, 'gender':'FEMALE', 'spoken_style':'독백체', 'emotion_style':'자신하는', 'emotion':'기쁨', 'intensity':3}
    caption=Custom_cap_function_fast(caption)

    infer_ctx = torch.inference_mode() if USE_INFERENCE_MODE else torch.no_grad()
    with infer_ctx:
        for pcm_bytes in cosyvoice.inference_zero_shot_stream(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
            caption=caption,
        ):
            chunk_count += 1

            yield pcm_bytes



def load_model():
    # Check for reference audio
    if not os.path.exists(REFERENCE_AUDIO):
        logger.error(f"Reference audio not found: {REFERENCE_AUDIO}", exc_info=True)
        return
    
    # Create output directory
    #Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load prompt_text from txt file next to audio
    prompt_text = load_prompt_text(REFERENCE_AUDIO, INSTRUCTION)
    
    print(f"\n🎤 Reference audio: {REFERENCE_AUDIO}")
    print(f"📝 Texts for synthesis: {len(SYNTHESIS_TEXTS)}")
    
    # Load model with parallel pipeline and TensorRT
    print("\n🔧 Loading FastCosyVoice3...")
    print(f"   - TensorRT Flow: {'✅' if USE_TRT_FLOW else '❌'}")
    print(f"   - TensorRT-LLM:  {'✅' if USE_TRT_LLM else '❌'} (dtype={TRT_LLM_DTYPE})")
    
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
    
    if USE_TRT_LLM and cosyvoice.trt_llm_loaded:
        print("✅ TensorRT-LLM loaded successfully")
    elif USE_TRT_LLM:
        print("⚠️ TensorRT-LLM not loaded, using PyTorch")
    
    # dtype diagnostics
    llm_dtype = next(cosyvoice.model.llm.parameters()).dtype
    flow_dtype = next(cosyvoice.model.flow.parameters()).dtype
    hift_dtype = next(cosyvoice.model.hift.parameters()).dtype
    print(f"📊 LLM dtype: {llm_dtype}, Flow dtype: {flow_dtype}, HiFT dtype: {hift_dtype}")
    
    sample_rate = cosyvoice.sample_rate
    print(f"📊 Sample rate: {sample_rate} Hz")
    
    # Parallel pipeline information
    print("\n🚀 Parallel pipeline:")
    if USE_TRT_LLM and cosyvoice.trt_llm_loaded:
        print("   - LLM: TensorRT-LLM (~3x speedup)")
    else:
        print("   - LLM: PyTorch + torch.compile")
    if USE_TRT_FLOW:
        print("   - Flow: TensorRT (~2.5x speedup)")
    else:
        print("   - Flow: PyTorch")
    print("   - Hift: PyTorch (f0_predictor on CPU)")
    
    # Apply torch.compile to LLM only if TRT-LLM is not used
    #if not (USE_TRT_LLM and cosyvoice.trt_llm_loaded):
    #    print("\n⚡ Applying torch.compile to LLM...")
    #    compile_start = time.time()
    #    apply_torch_compile(cosyvoice)
    #    compile_time = time.time() - compile_start
    #    print(f"✅ torch.compile applied in {compile_time:.3f} sec")
    #else:
    #    print("\n⚡ torch.compile skipped (using TensorRT-LLM)")
    
    # Prepare speaker embeddings (once)
    print("\n🎯 Preparing speaker embeddings...")
    spk_id = "reference_speaker"
    embed_start = time.time()
    cosyvoice.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, spk_id)
    embed_time = time.time() - embed_start
    print(f"✅ Embeddings prepared in {embed_time:.3f} sec")
    
    # Model warmup
    #if USE_TRT_LLM and cosyvoice.trt_llm_loaded:
    #    # With TRT-LLM warmup is shorter - only Flow and Hift
    #    print("\n🔥 Warming up model (TRT-LLM doesn't require long warmup)...")
    #    for _ in cosyvoice.inference_zero_shot_stream(
    #        tts_text="Short model warmup.",
    #        prompt_text=prompt_text,
    #        prompt_wav=REFERENCE_AUDIO,
    #        zero_shot_spk_id=spk_id,
    #    ):
    #        pass
    #    if torch.cuda.is_available():
    #        torch.cuda.synchronize()
    #    print("✅ Model warmed up")
    #else:
    #    # Without TRT-LLM full warmup is needed for torch.compile
    #    print("\n🔥 Warming up model (compiling graphs for different text lengths)...")
    #    warmup_model(cosyvoice, prompt_text, spk_id)
    #    print("✅ Model warmed up and ready")

    return cosyvoice, prompt_text, spk_id, sample_rate


def load_model_basic():
    # Check for reference audio
    if not os.path.exists(REFERENCE_AUDIO):
        logger.error(f"Reference audio not found: {REFERENCE_AUDIO}", exc_info=True)
        return
    
    # Create output directory
    #Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load prompt_text from txt file next to audio
    prompt_text = load_prompt_text(REFERENCE_AUDIO, INSTRUCTION)
    
    print(f"\n🎤 Reference audio: {REFERENCE_AUDIO}")
    print(f"📝 Texts for synthesis: {len(SYNTHESIS_TEXTS)}")
    
    load_start = time.time()
    
    cosyvoice = CosyVoice3(
        model_dir=MODEL_DIR,
        fp16=False,
        load_vllm=False,
        load_trt=False,
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
    
    
    # Prepare speaker embeddings (once)
    print("\n🎯 Preparing speaker embeddings...")
    spk_id = "reference_speaker"
    embed_start = time.time()
    cosyvoice.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, spk_id)
    embed_time = time.time() - embed_start
    print(f"✅ Embeddings prepared in {embed_time:.3f} sec")
    
    # Warmup run to initialize CUDA kernels and allocate memory
    print("\n🔥 Warmup run...")
    warmup_text = "Тестовый прогрев модели."
    warmup_start = time.time()
    for _ in cosyvoice.inference_zero_shot(
        tts_text=warmup_text,
        prompt_text=prompt_text,
        prompt_wav=REFERENCE_AUDIO,
        zero_shot_spk_id=spk_id,
        stream=True,
    ):
        pass  # Just iterate through to trigger computation

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start

    return cosyvoice, prompt_text, spk_id, sample_rate


if __name__ == '__main__':
    pass

