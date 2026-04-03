#!/usr/bin/env python3
"""
Example: multi-process TTS (LLM and Flow+HiFT in separate processes).

Usage:
    python run_mp_tts.py
"""

import logging
import os
import sys
import time
import wave
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.append("third_party/Matcha-TTS")

from fastcosyvoice.mp_tts import MultiProcessTTS

torch.set_float32_matmul_precision("high")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Load config ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_cfg = OmegaConf.load(_PROJECT_ROOT / "config.yaml")

MODEL_DIR      = _cfg["model"]["model_dir"]
LLM_PT_PATH    = _cfg["model"]["llm_checkpoint"]
FLOW_PT_PATH   = _cfg["model"]["flow_checkpoint"]
HIFT_PT_PATH   = _cfg["model"]["hift_checkpoint"]

# ── Speaker definitions (from config.yaml "아이유") ──────────────────────
_iu_cfg = next(s for s in _cfg["speakers"] if s["name"] == "아이유")
_iu_audio = Path(_iu_cfg["audio"])
_iu_prompt_txt = _iu_audio.with_suffix(".txt")
SPEAKERS = {
    "iu": {
        "audio": str(_iu_audio),
        "prompt": _iu_prompt_txt.read_text(encoding="utf-8").strip(),
    },
}
CAPTION     = '"약하게" "기쁜" 감정이고 "독백체" 스타일<|endofprompt|>'

OUTPUT_DIR     = "output/mp_tts_test_multi_v3"

# ── Batch: (text, speaker_id) ────────────────────────────────────────────
BATCH_ITEMS = [
    ("안녕하세요. 배치 스트리밍 테스트 첫 번째 문장입니다. I'm all alone, the rooms are getting smaller.", "iu"),
    ("An empty street, an empty house a hole inside my heart. I'm all alone, the rooms are getting smaller.", "iu"),
    ("안녕하세요, 이천이십육년 십이월 이십삼일 금요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다.", "iu"),
    ("어느덧 한 주를 마무리하는 시간인데, 다들 오늘 하루는 어떻게 보내셨나요?", "iu"),
    ("창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다.", "iu"),
    ("기술은 빠르게 변하고 세상은 복잡해졌지만, 좋은 음악이 주는 그 변함없는 울림은 언제나 우리를 안심하게 하죠.", "iu"),
    ("오늘은 특별히 이번 주 여러분이 나누어 주신 이야기들 속에서 느낀 평온함이라는 키워드로 플레이리스트를 준비했습니다.", "iu"),
    ("데이터가 계산한 선곡이라기보다는, 여러분의 감정선에 조용히 주파수를 맞춘 결과물이라고 할 수 있겠네요.", "iu"),
    ("차가운 일월의 공기를 녹여줄 수 있는, 포근한 담요 같은 노래들을 함께 들으려 합니다.", "iu"),
    ("잠시 일상의 소음은 뒤로하고, 지금 이 순간 흐르는 선율에 오롯이 몸을 맡겨보시는 건 어떨까요?", "iu"),
    ("창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다.", "iu"),
    ("기술은 빠르게 변하고 세상은 복잡해졌지만, 좋은 음악이 주는 그 변함없는 울림은 언제나 우리를 안심하게 하죠.", "iu"),
    ("오늘은 특별히 이번 주 여러분이 나누어 주신 이야기들 속에서 느낀 평온함이라는 키워드로 플레이리스트를 준비했습니다.", "iu"),
    ("데이터가 계산한 선곡이라기보다는, 여러분의 감정선에 조용히 주파수를 맞춘 결과물이라고 할 수 있겠네요.", "iu"),
    ("차가운 일월의 공기를 녹여줄 수 있는, 포근한 담요 같은 노래들을 함께 들으려 합니다.", "iu"),
    ("잠시 일상의 소음은 뒤로하고, 지금 이 순간 흐르는 선율에 오롯이 몸을 맡겨보시는 건 어떨까요?", "iu"),
]

BATCH_TEXTS   = [item[0] for item in BATCH_ITEMS]
BATCH_SPK_IDS = [item[1] for item in BATCH_ITEMS]


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def warmup(model: "MultiProcessTTS", spk_id: str, num_passes: int = 1) -> None:
    """Warm up TRT / CUDA kernels with various text lengths and batch sizes.

    Covers short → long single texts and a small batch so that TRT/cuDNN
    sees the full range of input shapes before real inference begins.
    """
    # 길이별 단일 텍스트 (짧은 → 긴 순서)
    warmup_texts_single = [
        # 짧은 (~20자)
        "안녕하세요, 반갑습니다.",
        # 중간 (~60자)
        "오늘 하루도 수고 많으셨습니다. 편안한 저녁 되세요.",
        # 긴 (~120자)
        "안녕하세요, 이천이십육년 삼월 이십이일 일요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다. 오늘도 좋은 음악과 함께해요.",
        # 아주 긴 (~200자)
        "기술은 빠르게 변하고 세상은 복잡해졌지만, 좋은 음악이 주는 그 변함없는 울림은 언제나 우리를 안심하게 하죠.",
        "창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다.",
    ]

    # 배치 텍스트 (실제 추론과 유사한 배치 크기)
    warmup_texts_batch = [
        # 짧은 (~20자)
        "안녕하세요, 반갑습니다.",
        # 중간 (~60자)
        "오늘 하루도 수고 많으셨습니다. 편안한 저녁 되세요.",
        # 긴 (~120자)
        "안녕하세요, 이천이십육년 삼월 이십이일 일요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다. 오늘도 좋은 음악과 함께해요.",
        # 아주 긴 (~200자)
        "사랑하는 사람과 함꼐 하는 음악 토크쇼, 가정의 평화를 지키기 위해 노력하는 당신의 모십이 너무 보기 좋습니다.",
    ]*4

    warmup_start = time.time()

    for pass_idx in range(num_passes):
        label = "kernel compilation" if pass_idx == 0 else "stabilization"
        logger.info("Warmup pass %d/%d (%s)...", pass_idx + 1, num_passes, label)

        # 단일 텍스트 (길이별)
        for i, text in enumerate(warmup_texts_single):
            logger.info("  [single] %d/%d  len=%d chars", i + 1, len(warmup_texts_single), len(text))
            for _ in model.inference_zero_shot_stream(
                tts_text=text,
                zero_shot_spk_id=spk_id,
            ):
                pass

        # 배치 텍스트
        logger.info("  [batch]  batch_size=%d", len(warmup_texts_batch))
        for _ in model.inference_zero_shot_stream(
            tts_text=warmup_texts_batch,
            zero_shot_spk_id=spk_id,
        ):
            pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info("Warmup finished in %.2f sec", time.time() - warmup_start)


def main():
    # Verify all speaker audio files exist
    for spk_id, spk_info in SPEAKERS.items():
        if not os.path.exists(spk_info["audio"]):
            logger.error("Reference audio not found for speaker '%s': %s", spk_id, spk_info["audio"])
            return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Initialise (spawns LLM + Flow processes) ──────────────────────────
    load_start = time.time()
    model = MultiProcessTTS(
        model_dir=MODEL_DIR,
        llm_pt_path=LLM_PT_PATH,
        flow_pt_path=FLOW_PT_PATH,
        hift_pt_path=HIFT_PT_PATH,
        fp16=True,
        load_trt=True,
        flow_trt_max_batch_size=8,
        llm_device='cuda:0',
        flow_devices=['cuda:0','cuda:0'],  # 2 Flow procs, each batch=8 → LLM batch=16
    )
    logger.info("Processes started in %.2f sec", time.time() - load_start)

    # ── Register all speakers ─────────────────────────────────────────────
    for spk_id, spk_info in SPEAKERS.items():
        model.add_zero_shot_spk(spk_info["prompt"], spk_info["audio"], spk_id)
        logger.info("Speaker registered: %s", spk_id)

    # ── Warmup (use first speaker) ────────────────────────────────────────
    first_spk_id = next(iter(SPEAKERS))
    warmup(model, first_spk_id)

    # ── Batch streaming inference (multi-speaker) ─────────────────────────
    audio_chunks: dict = {}
    first_chunk_time: dict = {}   # 샘플별 첫 청크 시각
    last_chunk_time: dict = {}    # 샘플별 마지막 청크 시각
    start = time.time()

    for chunk in model.inference_zero_shot_stream(
        tts_text=BATCH_TEXTS,
        zero_shot_spk_ids=BATCH_SPK_IDS,
    ):
        now = time.time()
        if isinstance(chunk, dict):
            idx = chunk["sample_idx"]
            pcm = chunk["pcm_bytes"]
        else:
            idx = 0
            pcm = chunk
        audio_chunks.setdefault(idx, b"")
        audio_chunks[idx] += pcm
        if idx not in first_chunk_time:
            first_chunk_time[idx] = now
        last_chunk_time[idx] = now

    total = time.time() - start
    logger.info("Batch done in %.2f sec", total)

    # ── Save outputs ──────────────────────────────────────────────────────
    total_audio = 0.0
    for idx, pcm in sorted(audio_chunks.items()):
        out = os.path.join(OUTPUT_DIR, f"sample_{idx:02d}.wav")
        write_wav(out, pcm, model.sample_rate)
        duration = len(pcm) / 2 / model.sample_rate
        total_audio += duration
        # 샘플별 실제 소요 시간: 첫 청크 ~ 마지막 청크
        ttfb = first_chunk_time[idx] - start                       # time to first byte
        sample_elapsed = last_chunk_time[idx] - first_chunk_time[idx]  # 생성 구간
        sample_rtf = sample_elapsed / duration if duration > 0 else float("inf")
        logger.info(
            "sample %d (%s): audio=%.3fs  ttfb=%.3fs  elapsed=%.3fs  rtf=%.3f  -> %s",
            idx, BATCH_SPK_IDS[idx], duration, ttfb, sample_elapsed, sample_rtf, out,
        )
    logger.info("Batch RTF = %.4f  (%.2fs wall / %.2fs audio)", total / total_audio, total, total_audio)

    model.stop()


if __name__ == "__main__":
    # IMPORTANT: 'spawn' is required for CUDA in child processes.
    # MultiProcessTTS uses mp.get_context('spawn') internally so this is
    # not strictly required here, but it prevents accidental fork on Linux.
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
