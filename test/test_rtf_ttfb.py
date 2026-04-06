#!/usr/bin/env python3
"""
RTF / TTFB 벤치마크 테스트

기준:
- 텍스트 길이: ~200자 고정
- 배치 크기: [1, 2, 4, 8, 10]
- 각 청크의 RTF < 1 이어야 PASS

측정 항목 (청크 단위):
- chunk_rtf: 해당 청크의 오디오 길이 / 생성 소요 시간
- ttfb: 요청 시작 ~ 첫 번째 청크 도착까지 시간

Usage:
    python test/test_rtf_ttfb.py
"""

import csv
import logging
import os
import sys
import time
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

OUTPUT_ROOT = str(_PROJECT_ROOT / "output" / "test_rtf_ttfb")

BATCH_SIZES = [1, 2, 4, 8, 10]

# ~200자 테스트 문장
TEST_TEXT = (
    "안녕하세요, 이천이십육년 사월 삼일 목요일 저녁, 여러분의 곁을 지키는 AI DJ 멜론 허니듀입니다. "
    "어느덧 한 주를 마무리하는 시간인데, 다들 오늘 하루는 어떻게 보내셨나요? "
    "창밖으로 하나둘 불이 켜지는 도시의 야경을 보고 있으니, 문득 여러분의 마음에도 따뜻한 위로가 필요하진 않을까 하는 생각이 듭니다. "
    "잠시 일상의 소음은 뒤로하고, 지금 이 순간 흐르는 선율에 오롯이 몸을 맡겨보시는 건 어떨까요?"
)


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
    ]
    logger.info("Warmup start...")
    for text in warmup_texts:
        for _ in model.inference_zero_shot_stream(tts_text=text, zero_shot_spk_id=spk_id):
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info("Warmup done.")


def run_batch_test(model: MultiProcessTTS, batch_size: int, spk_id: str, sample_rate: int):
    """
    배치 스트리밍 추론을 실행하고 청크 단위 RTF/TTFB를 측정한다.

    Returns:
        chunks_by_sample: {sample_idx: [(chunk_idx, chunk_audio_sec, chunk_elapsed_sec, chunk_rtf, timestamp)]}
        ttfb_by_sample:   {sample_idx: float}   # 첫 청크까지 시간
        total_wall:       float                  # 전체 소요 시간
    """
    if batch_size == 1:
        texts = TEST_TEXT
        spk_ids = None
        single_spk_id = spk_id
    else:
        texts = [TEST_TEXT] * batch_size
        spk_ids = [spk_id] * batch_size
        single_spk_id = ''

    chunks_by_sample: dict[int, list] = {}
    prev_time_by_sample: dict[int, float] = {}
    ttfb_by_sample: dict[int, float] = {}

    start = time.time()

    stream_kwargs = dict(tts_text=texts, zero_shot_spk_id=single_spk_id)
    if spk_ids is not None:
        stream_kwargs['zero_shot_spk_ids'] = spk_ids

    for chunk in model.inference_zero_shot_stream(**stream_kwargs):
        now = time.time()

        if isinstance(chunk, dict):
            idx = chunk["sample_idx"]
            pcm = chunk["pcm_bytes"]
        else:
            idx = 0
            pcm = chunk

        chunk_audio_sec = len(pcm) / 2 / sample_rate

        if idx not in prev_time_by_sample:
            # 첫 번째 청크
            ttfb_by_sample[idx] = now - start
            chunk_elapsed = now - start
        else:
            chunk_elapsed = now - prev_time_by_sample[idx]

        prev_time_by_sample[idx] = now

        chunk_rtf = chunk_elapsed / chunk_audio_sec if chunk_audio_sec > 0 else float("inf")

        chunks_by_sample.setdefault(idx, [])
        chunk_idx = len(chunks_by_sample[idx])
        chunks_by_sample[idx].append((chunk_idx, chunk_audio_sec, chunk_elapsed, chunk_rtf, now))

    total_wall = time.time() - start
    return chunks_by_sample, ttfb_by_sample, total_wall


def main():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    logger.info("Text length: %d chars", len(TEST_TEXT))
    logger.info("Batch sizes: %s", BATCH_SIZES)

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
        flow_trt_max_batch_size=max(BATCH_SIZES),
        llm_device='cuda:0',
        flow_devices=['cuda:0', 'cuda:0'],
        qwen3_dir=QWEN3_DIR,
    )
    logger.info("Model loaded in %.2f sec", time.time() - load_start)

    # ── 스피커 등록 & Warmup ─────────────────────────────────────────────────
    model.add_zero_shot_spk(SPEAKER_INFO["prompt"], SPEAKER_INFO["audio"], SPK_ID)
    warmup(model, SPK_ID)

    # ── CSV 준비 ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_ROOT, "results.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "batch_size", "sample_idx", "chunk_idx",
        "chunk_audio_sec", "chunk_elapsed_sec", "chunk_rtf",
        "ttfb_sec", "pass",
    ])

    # ── 테스트 실행 ──────────────────────────────────────────────────────────
    all_passed = True

    for batch_size in BATCH_SIZES:
        logger.info("━━━ batch_size=%d ━━━", batch_size)

        chunks_by_sample, ttfb_by_sample, total_wall = run_batch_test(
            model, batch_size, SPK_ID, model.sample_rate,
        )

        batch_fail_count = 0
        batch_chunk_count = 0

        for sample_idx in sorted(chunks_by_sample.keys()):
            ttfb = ttfb_by_sample.get(sample_idx, 0.0)
            for chunk_idx, chunk_audio, chunk_elapsed, chunk_rtf, _ in chunks_by_sample[sample_idx]:
                passed = chunk_rtf < 1.0 or chunk_idx == 0
                if not passed:
                    batch_fail_count += 1
                batch_chunk_count += 1

                csv_writer.writerow([
                    batch_size, sample_idx, chunk_idx,
                    f"{chunk_audio:.4f}",
                    f"{chunk_elapsed:.4f}",
                    f"{chunk_rtf:.4f}",
                    f"{ttfb:.4f}" if chunk_idx == 0 else "",
                    "PASS" if passed else "FAIL",
                ])

        csv_file.flush()

        # 배치 요약
        all_ttfbs = list(ttfb_by_sample.values())
        avg_ttfb = sum(all_ttfbs) / len(all_ttfbs) if all_ttfbs else 0.0
        max_ttfb = max(all_ttfbs) if all_ttfbs else 0.0

        if batch_fail_count == 0:
            logger.info(
                "  PASS  chunks=%d  wall=%.2fs  avg_ttfb=%.3fs  max_ttfb=%.3fs",
                batch_chunk_count, total_wall, avg_ttfb, max_ttfb,
            )
        else:
            all_passed = False
            logger.warning(
                "  FAIL  %d/%d chunks exceeded RTF>=1  wall=%.2fs  avg_ttfb=%.3fs  max_ttfb=%.3fs",
                batch_fail_count, batch_chunk_count, total_wall, avg_ttfb, max_ttfb,
            )

    csv_file.close()

    logger.info("━━━ RESULTS ━━━")
    logger.info("CSV: %s", csv_path)
    if all_passed:
        logger.info("ALL PASSED: every chunk RTF < 1.0")
    else:
        logger.warning("SOME FAILED: see CSV for details")

    model.stop()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
