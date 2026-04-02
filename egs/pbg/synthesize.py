#!/usr/bin/env python3
"""
Stage 1: 레퍼런스 오디오로부터 zero-shot TTS 합성
- filelist.txt 의 각 레퍼런스 오디오에 대해 합성 수행
- 출력: egs/pbg/output/{파일명}.wav
"""

import argparse
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "third_party" / "Matcha-TTS"))

from fastcosyvoice.mp_tts import MultiProcessTTS

torch.set_float32_matmul_precision("high")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TTS_TEXT = "안녕하세요. 오늘 날씨가 정말 좋네요. 함께 산책하러 가실래요?"


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def convert_mp3_to_wav(mp3_path: str) -> str:
    """MP3를 16kHz WAV로 변환하여 같은 경로에 저장."""
    wav_path = mp3_path.replace(".mp3", ".wav")
    if not os.path.exists(wav_path):
        waveform, sr = torchaudio.load(mp3_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        torchaudio.save(wav_path, waveform, 16000)
    return wav_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, default="egs/pbg/filelist.txt")
    parser.add_argument("--data_dir", type=str, default="egs/pbg/data")
    parser.add_argument("--output_dir", type=str, default="egs/pbg/output")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--load_trt", action="store_true", default=False)
    args = parser.parse_args()

    # Load config
    project_root = Path(__file__).resolve().parents[2]
    with open(project_root / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_dir = str(project_root / cfg["model"]["model_dir"])
    llm_pt = str(project_root / cfg["model"]["llm_checkpoint"])
    flow_pt = str(project_root / cfg["model"]["flow_checkpoint"])
    hift_pt = str(project_root / cfg["model"]["hift_checkpoint"])
    qwen3_dir = str(project_root / cfg["model"]["qwen3_dir"])

    # Load filelist
    with open(args.filelist, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip()]

    logger.info("Total files to synthesize: %d", len(filenames))

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (MultiProcessTTS uses cosyvoice3_xvec.yaml internally)
    logger.info("Loading model...")
    load_start = time.time()
    model = MultiProcessTTS(
        model_dir=model_dir,
        llm_pt_path=llm_pt,
        flow_pt_path=flow_pt,
        hift_pt_path=hift_pt,
        fp16=args.fp16,
        load_trt=args.load_trt,
        llm_device="cuda:0",
        flow_devices=["cuda:0"],
        qwen3_dir=qwen3_dir,
    )
    logger.info("Model loaded in %.2f sec", time.time() - load_start)

    data_dir = Path(args.data_dir)
    success = 0
    fail = 0

    for i, name in enumerate(filenames):
        mp3_path = str(data_dir / f"{name}.mp3")
        json_path = str(data_dir / f"{name}.json")

        if not os.path.exists(mp3_path) or not os.path.exists(json_path):
            logger.warning("Skipping %s: file not found", name)
            fail += 1
            continue

        # Read prompt text from JSON
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        prompt_text = meta["text"].strip()

        if not prompt_text:
            logger.warning("Skipping %s: empty prompt text", name)
            fail += 1
            continue

        # Convert MP3 to WAV for reference audio
        wav_path = convert_mp3_to_wav(mp3_path)

        # Register speaker for this reference audio
        spk_id = name
        try:
            model.add_zero_shot_spk(prompt_text, wav_path, spk_id)
        except Exception as e:
            logger.error("Failed to register speaker %s: %s", name, e)
            fail += 1
            continue

        # Synthesize
        out_path = str(output_dir / f"{name}.wav")
        logger.info("[%d/%d] Synthesizing %s", i + 1, len(filenames), name)

        try:
            pcm_bytes = b""
            for chunk in model.inference_zero_shot_stream(
                tts_text=TTS_TEXT,
                zero_shot_spk_id=spk_id,
            ):
                pcm_bytes += chunk

            write_wav(out_path, pcm_bytes, model.sample_rate)
            duration = len(pcm_bytes) / 2 / model.sample_rate
            logger.info("  -> %s (%.2fs)", out_path, duration)
            success += 1
        except Exception as e:
            logger.error("  Failed %s: %s", name, e)
            fail += 1

    logger.info("Done. Success: %d, Failed: %d", success, fail)
    model.stop()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
