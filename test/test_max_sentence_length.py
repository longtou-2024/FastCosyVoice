#!/usr/bin/env python3
"""
최대 문장 길이 테스트 스크립트

배치 스트리밍 음성합성에서 지원 가능한 최대 문장 길이를 테스트합니다.
- 배치 크기: 10 (고정)
- 문장 길이: 150 ~ 400자
- 레퍼런스 오디오: 아이유 (고정)
- 측정 항목: RTF, TTFB
- 합성된 음성 파일과 텍스트를 함께 저장하여 수동 검증 가능

Usage:
    python test/test_max_sentence_length.py
"""

import csv
import logging
import os
import sys
import time
import traceback
import wave
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent / "third_party" / "Matcha-TTS"))

from fastcosyvoice.mp_tts import MultiProcessTTS

torch.set_float32_matmul_precision("high")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_cfg = OmegaConf.load(_PROJECT_ROOT / "config.yaml")

MODEL_DIR    = _cfg["model"]["model_dir"]
LLM_PT_PATH  = _cfg["model"]["llm_checkpoint"]
FLOW_PT_PATH = _cfg["model"]["flow_checkpoint"]
HIFT_PT_PATH = _cfg["model"]["hift_checkpoint"]
QWEN3_DIR    = _cfg["model"]["qwen3_dir"]

_iu_cfg = next(s for s in _cfg["speakers"] if s["name"] == "아이유")
_iu_audio = Path(_iu_cfg["audio"])
_iu_prompt_txt = _iu_audio.with_suffix(".txt")
SPEAKER_INFO = {
    "audio": str(_iu_audio),
    "prompt": _iu_prompt_txt.read_text(encoding="utf-8").strip(),
}
SPK_ID = "iu"

OUTPUT_ROOT = str(_PROJECT_ROOT / "output" / "test_max_sentence_length")

# ── 테스트 문장 정의 ────────────────────────────────────────────────────────
# 텍스트 길이별 테스트 문장: 150, 200, 250, 300, 350, 400자
# 각 튜플: (라벨, 텍스트)

TEST_SENTENCES: list[tuple[str, str]] = [
    ("len_150", (
        "기술은 빠르게 변하고 세상은 복잡해졌지만, 좋은 음악이 주는 그 변함없는 울림은 언제나 우리를 안심하게 하죠. "
        "오늘은 특별히 이번 주 여러분이 나누어 주신 이야기들 속에서 느낀 평온함이라는 키워드로 플레이리스트를 준비했습니다. "
        "데이터가 계산한 선곡이라기보다는, 여러분의 감정선에 조용히 주파수를 맞춘 결과물이라고 할 수 있겠네요."
    )),
    ("len_200", (
        "안녕하세요, 이천이십육년 사월 삼일 목요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다. "
        "어느덧 한 주를 마무리하는 시간인데, 다들 오늘 하루는 어떻게 보내셨나요? "
        "창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다. "
        "잠시 일상의 소음은 뒤로하고, 지금 이 순간 흐르는 선율에 오롯이 몸을 맡겨보시는 건 어떨까요?"
    )),
    ("len_250", (
        "안녕하세요, 이천이십육년 사월 삼일 목요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다. "
        "어느덧 한 주를 마무리하는 시간인데, 다들 오늘 하루는 어떻게 보내셨나요? "
        "창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다. "
        "기술은 빠르게 변하고 세상은 복잡해졌지만, 좋은 음악이 주는 그 변함없는 울림은 언제나 우리를 안심하게 하죠. "
        "잠시 일상의 소음은 뒤로하고, 지금 이 순간 흐르는 선율에 오롯이 몸을 맡겨보시는 건 어떨까요?"
    )),
    ("len_300", (
        "안녕하세요, 이천이십육년 사월 삼일 목요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다. "
        "어느덧 한 주를 마무리하는 시간인데, 다들 오늘 하루는 어떻게 보내셨나요? "
        "창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다. "
        "기술은 빠르게 변하고 세상은 복잡해졌지만, 좋은 음악이 주는 그 변함없는 울림은 언제나 우리를 안심하게 하죠. "
        "오늘은 특별히 이번 주 여러분이 나누어 주신 이야기들 속에서 느낀 평온함이라는 키워드로 플레이리스트를 준비했습니다. "
        "데이터가 계산한 선곡이라기보다는, 여러분의 감정선에 조용히 주파수를 맞춘 결과물이라고 할 수 있겠네요."
    )),
    ("len_350", (
        "안녕하세요, 이천이십육년 사월 삼일 목요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다. "
        "어느덧 한 주를 마무리하는 시간인데, 다들 오늘 하루는 어떻게 보내셨나요? "
        "창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다. "
        "기술은 빠르게 변하고 세상은 복잡해졌지만, 좋은 음악이 주는 그 변함없는 울림은 언제나 우리를 안심하게 하죠. "
        "오늘은 특별히 이번 주 여러분이 나누어 주신 이야기들 속에서 느낀 평온함이라는 키워드로 플레이리스트를 준비했습니다. "
        "데이터가 계산한 선곡이라기보다는, 여러분의 감정선에 조용히 주파수를 맞춘 결과물이라고 할 수 있겠네요. "
        "차가운 사월의 공기를 녹여줄 수 있는, 포근한 담요 같은 노래들을 함께 들으려 합니다."
    )),
    ("len_400", (
        "안녕하세요, 이천이십육년 사월 삼일 목요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다. "
        "어느덧 한 주를 마무리하는 시간인데, 다들 오늘 하루는 어떻게 보내셨나요? "
        "창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다. "
        "기술은 빠르게 변하고 세상은 복잡해졌지만, 좋은 음악이 주는 그 변함없는 울림은 언제나 우리를 안심하게 하죠. "
        "오늘은 특별히 이번 주 여러분이 나누어 주신 이야기들 속에서 느낀 평온함이라는 키워드로 플레이리스트를 준비했습니다. "
        "데이터가 계산한 선곡이라기보다는, 여러분의 감정선에 조용히 주파수를 맞춘 결과물이라고 할 수 있겠네요. "
        "차가운 사월의 공기를 녹여줄 수 있는, 포근한 담요 같은 노래들을 함께 들으려 합니다. "
        "잠시 일상의 소음은 뒤로하고, 지금 이 순간 흐르는 선율에 오롯이 몸을 맡겨보시는 건 어떨까요?"
    )),
]


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def warmup(model: MultiProcessTTS, spk_id: str) -> None:
    """Warmup TRT/CUDA kernels."""
    warmup_texts = [
        "안녕하세요, 반갑습니다.",
        "오늘 하루도 수고 많으셨습니다. 편안한 저녁 되세요.",
        "안녕하세요, 이천이십육년 사월 삼일 목요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다.",
    ]
    logger.info("Warmup start...")
    for text in warmup_texts:
        for _ in model.inference_zero_shot_stream(
            tts_text=text,
            zero_shot_spk_id=spk_id,
        ):
            pass
    # 배치 warmup
    for _ in model.inference_zero_shot_stream(
        tts_text=warmup_texts * 2,
        zero_shot_spk_id=spk_id,
    ):
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info("Warmup done.")


def run_single_test(
    model: MultiProcessTTS,
    texts: list[str],
    spk_id: str,
) -> dict:
    """
    배치 스트리밍 추론을 실행하고 결과를 반환한다.

    Returns:
        {
            "success": bool,
            "error": str | None,
            "audio_chunks": {idx: bytes},
            "ttfb": {idx: float},          # 샘플별 Time To First Byte (초)
            "rtf": {idx: float},            # 샘플별 Real Time Factor
            "audio_duration": {idx: float}, # 샘플별 오디오 길이 (초)
            "wall_time": float,             # 전체 소요 시간
            "batch_rtf": float,             # 전체 RTF
        }
    """
    batch_size = len(texts)
    audio_chunks: dict[int, bytes] = {}
    first_chunk_time: dict[int, float] = {}
    last_chunk_time: dict[int, float] = {}

    try:
        if batch_size == 1:
            # 단일 텍스트
            start = time.time()
            for chunk in model.inference_zero_shot_stream(
                tts_text=texts[0],
                zero_shot_spk_id=spk_id,
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
        else:
            # 배치 텍스트
            spk_ids = [spk_id] * batch_size
            start = time.time()
            for chunk in model.inference_zero_shot_stream(
                tts_text=texts,
                zero_shot_spk_ids=spk_ids,
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

        wall_time = time.time() - start

    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "audio_chunks": audio_chunks,
            "ttfb": {},
            "rtf": {},
            "audio_duration": {},
            "wall_time": time.time() - start if 'start' in dir() else 0.0,
            "batch_rtf": 0.0,
        }

    # 결과 계산
    sample_rate = model.sample_rate
    ttfb_map = {}
    rtf_map = {}
    dur_map = {}
    total_audio = 0.0

    for idx, pcm in sorted(audio_chunks.items()):
        duration = len(pcm) / 2 / sample_rate
        dur_map[idx] = duration
        total_audio += duration

        if idx in first_chunk_time:
            ttfb_map[idx] = first_chunk_time[idx] - start
            sample_elapsed = last_chunk_time[idx] - first_chunk_time[idx]
            rtf_map[idx] = sample_elapsed / duration if duration > 0 else float("inf")
        else:
            ttfb_map[idx] = 0.0
            rtf_map[idx] = 0.0

    batch_rtf = wall_time / total_audio if total_audio > 0 else float("inf")

    return {
        "success": True,
        "error": None,
        "audio_chunks": audio_chunks,
        "ttfb": ttfb_map,
        "rtf": rtf_map,
        "audio_duration": dur_map,
        "wall_time": wall_time,
        "batch_rtf": batch_rtf,
    }


def main():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # ── 모델 초기화 ──────────────────────────────────────────────────────────
    logger.info("Loading model...")
    load_start = time.time()
    model = MultiProcessTTS(
        model_dir=MODEL_DIR,
        llm_pt_path=LLM_PT_PATH,
        flow_pt_path=FLOW_PT_PATH,
        hift_pt_path=HIFT_PT_PATH,
        fp16=True,
        load_trt=True,
        flow_trt_max_batch_size=10,
        llm_device='cuda:0',
        flow_devices=['cuda:0', 'cuda:0'],
        qwen3_dir=QWEN3_DIR,
    )
    logger.info("Model loaded in %.2f sec", time.time() - load_start)

    # ── 스피커 등록 ──────────────────────────────────────────────────────────
    model.add_zero_shot_spk(SPEAKER_INFO["prompt"], SPEAKER_INFO["audio"], SPK_ID)
    logger.info("Speaker registered: %s", SPK_ID)

    # ── Warmup ───────────────────────────────────────────────────────────────
    warmup(model, SPK_ID)

    # ── CSV 결과 파일 준비 ───────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_ROOT, "results.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "batch_size", "sentence_label", "sample_idx", "text_length",
        "audio_duration_sec", "ttfb_sec", "rtf", "wall_time_sec",
        "batch_rtf", "success", "error", "wav_path",
    ])

    # ── 테스트 실행 ──────────────────────────────────────────────────────────
    batch_sizes = [10]

    for batch_size in batch_sizes:
        for label, text in TEST_SENTENCES:
            test_name = f"batch{batch_size:02d}_{label}"
            test_dir = os.path.join(OUTPUT_ROOT, test_name)
            Path(test_dir).mkdir(parents=True, exist_ok=True)

            # 같은 문장을 batch_size 만큼 복제
            texts = [text] * batch_size
            text_len = len(text)

            logger.info(
                "━━━ TEST: %s | batch=%d, text_len=%d chars ━━━",
                test_name, batch_size, text_len,
            )

            result = run_single_test(model, texts, SPK_ID)

            if result["success"]:
                logger.info(
                    "  ✓ wall=%.2fs  batch_rtf=%.4f",
                    result["wall_time"], result["batch_rtf"],
                )
            else:
                logger.error("  ✗ ERROR: %s", result["error"])

            # 오디오 및 텍스트 저장
            for idx in range(batch_size):
                wav_path = os.path.join(test_dir, f"sample_{idx:02d}.wav")
                txt_path = os.path.join(test_dir, f"sample_{idx:02d}.txt")

                # 텍스트 저장 (합성 문장 확인용)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                # 오디오 저장
                pcm = result["audio_chunks"].get(idx, b"")
                if pcm:
                    write_wav(wav_path, pcm, model.sample_rate)
                else:
                    wav_path = ""  # 오디오 없음

                # CSV 기록
                csv_writer.writerow([
                    batch_size,
                    label,
                    idx,
                    text_len,
                    f"{result['audio_duration'].get(idx, 0.0):.3f}",
                    f"{result['ttfb'].get(idx, 0.0):.3f}",
                    f"{result['rtf'].get(idx, 0.0):.4f}",
                    f"{result['wall_time']:.3f}",
                    f"{result['batch_rtf']:.4f}",
                    result["success"],
                    result["error"] or "",
                    wav_path,
                ])
                csv_file.flush()

                if pcm:
                    dur = result["audio_duration"].get(idx, 0.0)
                    ttfb = result["ttfb"].get(idx, 0.0)
                    rtf = result["rtf"].get(idx, 0.0)
                    logger.info(
                        "  [%d] audio=%.2fs  ttfb=%.3fs  rtf=%.4f  -> %s",
                        idx, dur, ttfb, rtf, wav_path,
                    )

    csv_file.close()
    logger.info("━━━ ALL TESTS COMPLETE ━━━")
    logger.info("Results CSV: %s", csv_path)
    logger.info("Audio files: %s", OUTPUT_ROOT)

    model.stop()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
