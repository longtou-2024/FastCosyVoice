"""FastAPI streaming TTS server for FastCosyVoice.

Serves a web UI (index.html) and provides streaming TTS via MultiProcessTTS.
Designed for Google Cloud Run deployment (single sentence, batch_size=1).

Usage:
    python cloud_run/fastapi_tts_server.py
    python cloud_run/fastapi_tts_server.py --config config.yaml --port 7860
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import yaml
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Ensure third_party is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "third_party" / "Matcha-TTS"))
sys.path.insert(0, str(_PROJECT_ROOT / "KENT-G2P"))

from fastcosyvoice.mp_tts import MultiProcessTTS
from kent_g2p.CoreaSpeech.normalization import N2gkPlus
from kent_g2p.pron_trans.transliterator import PronTransliterator

torch.set_float32_matmul_precision("high")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FastCosyVoice TTS")

# ── Global state (set during startup) ─────────────────────────────────────
model: MultiProcessTTS = None
speakers: dict = {}  # {spk_id: {"audio": path, "prompt": text}}
default_caption: str = ""
model_status: str = "initializing"  # → "copying" → "loading" → "warming_up" → "ready" | "error"
model_error: str = ""

# ── Text preprocessing ────────────────────────────────────────────────────
text_normalizer = N2gkPlus()
pron_transliterator: PronTransliterator = None
pron_dic_version: str = ""


def _init_pron_transliterator(dic_dir: str):
    global pron_transliterator
    logger.info("PronTransliterator initializing (dic_dir=%s) ...", dic_dir)
    pron_transliterator = PronTransliterator(dic_dir=dic_dir)
    logger.info("PronTransliterator ready")


def _log_tmp_usage(label: str = ""):
    """Log /tmp disk (=memory on tmpfs) usage."""
    try:
        import shutil
        usage = shutil.disk_usage("/tmp")
        used_gb = (usage.total - usage.free) / (1024**3)
        total_gb = usage.total / (1024**3)
        logger.info("/tmp usage [%s]: %.1fGB / %.1fGB", label, used_gb, total_gb)
    except Exception:
        pass


def _log_memory(label: str = ""):
    """Log system memory usage."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                info[parts[0].rstrip(":")] = int(parts[1])  # kB
            total = info["MemTotal"] / (1024 * 1024)
            avail = info["MemAvailable"] / (1024 * 1024)
            used = total - avail
            logger.info("Memory [%s]: %.1fGB used / %.1fGB total (%.1fGB free)", label, used, total, avail)
    except Exception:
        pass


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def copy_from_gcs(gcs_path: str, local_dir: str):
    """Copy model files from GCS to local disk."""
    global model_status
    model_status = "copying"
    logger.info("Copying from GCS: %s → %s", gcs_path, local_dir)
    os.makedirs(local_dir, exist_ok=True)

    start = time.time()
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", gcs_path + "/*", local_dir + "/"],
        capture_output=False,
    )
    elapsed = time.time() - start
    logger.info("GCS copy exit=%d, elapsed=%.0fs", result.returncode, elapsed)

    if result.returncode != 0:
        raise RuntimeError(f"GCS copy failed with exit code {result.returncode}")

    _log_tmp_usage("After GCS copy")

    # Verify
    for f in ["config.yaml", "data/models", "data/checkpoints_v2", "data/checkpoints", "data/speakers"]:
        full = os.path.join(local_dir, f)
        status = "OK" if os.path.exists(full) else "MISSING"
        logger.info("  %s: %s", status, f)


def init_pipeline(config_path: str):
    """Full initialization: GCS copy (if needed) + model load + warmup. Runs in background thread."""
    global model, speakers, default_caption, model_status, model_error

    try:
        # GCS copy if configured
        gcs_path = os.environ.get("GCS_MODEL_PATH", "")
        if gcs_path:
            copy_from_gcs(gcs_path, "/tmp/model")
            config_path = "/tmp/model/config.yaml"

        project_root = Path(config_path).resolve().parent
        logger.info("Config path: %s (project_root=%s)", config_path, project_root)

        # Verify key files exist
        for f in ["config.yaml", "data/models", "data/checkpoints_v2", "data/speakers"]:
            full = project_root / f
            status = "OK" if full.exists() else "MISSING"
            logger.info("  %s: %s", status, f)

        cfg = load_config(config_path)
        logger.info("Config loaded successfully")

        # Parse pron_dic config (initialization deferred until after model loading)
        global pron_dic_version
        pron_dic = cfg.get("pron_dic", "")
        pron_dic_path = str(project_root / pron_dic) if pron_dic else ""
        import re as _re
        date_match = _re.search(r'/(\d{8})/', pron_dic)
        pron_dic_version = date_match.group(1) if date_match else ""

        model_status = "loading"
        model_dir = str(project_root / cfg["model"]["model_dir"])
        llm_pt = str(project_root / cfg["model"]["llm_checkpoint"])
        flow_pt = str(project_root / cfg["model"]["flow_checkpoint"])
        hift_pt = str(project_root / cfg["model"]["hift_checkpoint"])
        qwen3_dir = str(project_root / cfg["model"]["qwen3_dir"])

        # Cleanup callbacks: delete files from /tmp after they're loaded into memory
        import shutil

        def _on_frontend_loaded():
            """Frontend loaded Qwen3 + ONNX into memory. Delete from disk."""
            if not gcs_path:
                return
            for p in [qwen3_dir,
                      os.path.join(model_dir, 'campplus.onnx'),
                      os.path.join(model_dir, 'speech_tokenizer_v3.onnx'),
                      os.path.join(model_dir, 'spk2info.pt')]:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.isfile(p):
                    os.remove(p)
            _log_tmp_usage("After frontend cleanup")

        def _on_workers_ready():
            """LLM/Flow/HiFT loaded into GPU. Delete remaining model files."""
            if not gcs_path:
                return
            for p in [llm_pt, flow_pt, hift_pt]:
                if os.path.isfile(p):
                    os.remove(p)
            # Delete TRT plans and other model dir files
            for f in os.listdir(model_dir) if os.path.isdir(model_dir) else []:
                fp = os.path.join(model_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            _log_tmp_usage("After workers cleanup")

        logger.info("Loading model from %s ...", model_dir)
        load_start = time.time()
        model = MultiProcessTTS(
            model_dir=model_dir,
            llm_pt_path=llm_pt,
            flow_pt_path=flow_pt,
            hift_pt_path=hift_pt,
            fp16=True,
            load_trt=True,
            flow_trt_max_batch_size=8,
            llm_device="cuda:0",
            flow_devices=["cuda:0"],
            qwen3_dir=qwen3_dir,
            on_frontend_loaded=_on_frontend_loaded,
            on_workers_ready=_on_workers_ready,
        )
        logger.info("Model loaded in %.2f sec", time.time() - load_start)

        # Step 4: Register speakers
        skip_speakers = {"장원영_가이드"}
        for spk_cfg in cfg.get("speakers", []):
            name = spk_cfg["name"]
            if name in skip_speakers:
                logger.info("Skipping speaker: %s", name)
                continue
            audio_path = project_root / spk_cfg["audio"]
            prompt_txt_path = audio_path.with_suffix(".txt")

            if not audio_path.exists():
                logger.warning("Speaker audio not found, skipping: %s", audio_path)
                continue
            if not prompt_txt_path.exists():
                logger.warning("Speaker prompt txt not found, skipping: %s", prompt_txt_path)
                continue

            prompt = prompt_txt_path.read_text(encoding="utf-8").strip()
            spk_id = name
            model.add_zero_shot_spk(prompt, str(audio_path), spk_id)
            speakers[spk_id] = {"audio": str(audio_path), "prompt": prompt}
            logger.info("Speaker registered: %s", spk_id)

        default_caption = '"약하게" "기쁜" 감정이고 "독백체" 스타일<|endofprompt|>'

        # Free Qwen3 model — only needed during speaker registration
        _log_memory("Before Qwen3 cleanup")
        del model.frontend.Qwen3
        del model.frontend.tokenizer_q_3
        import gc; gc.collect()
        _log_memory("After Qwen3 cleanup")

        # Final cleanup: delete speaker files and anything remaining
        if gcs_path:
            shutil.rmtree("/tmp/model", ignore_errors=True)
            _log_tmp_usage("After final cleanup")

        # Initialize text preprocessor (after cleanup to avoid memory peak overlap)
        _init_pron_transliterator(pron_dic_path)

        # Step 5: Warmup
        model_status = "warming_up"
        if speakers:
            first_spk = next(iter(speakers))
            logger.info("Warming up with speaker '%s' ...", first_spk)
            warmup_start = time.time()
            for text in [
                "안녕하세요, 반갑습니다.",
                "오늘 하루도 수고 많으셨습니다. 편안한 저녁 되세요.",
            ]:
                for _ in model.inference_zero_shot_stream(
                    tts_text=text,
                    zero_shot_spk_id=first_spk,
                ):
                    pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("Warmup done in %.2f sec", time.time() - warmup_start)

        model_status = "ready"
        _log_memory("Model ready")
        _log_tmp_usage("Model ready")
        logger.info("Model ready!")

    except Exception as e:
        model_status = "error"
        model_error = str(e)
        logger.exception("Model initialization failed")


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
async def health():
    return {"status": model_status, "error": model_error}


@app.get("/api/speakers")
async def get_speakers():
    return {"speakers": list(speakers.keys()), "status": model_status}


@app.post("/api/tts/stream")
async def tts_stream(payload: dict):
    """Streaming TTS endpoint. Returns chunked PCM audio (int16, 24kHz, mono).

    Request body: { "text": "...", "speaker": "아이유" }

    Appends a JSON metadata trailer at the end (after 4 zero-byte marker).
    """
    if model_status != "ready":
        return JSONResponse(status_code=503, content={"error": f"Model not ready (status: {model_status})"})

    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text is required."})

    spk_id = payload.get("speaker", "")
    if spk_id not in speakers:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown speaker: {spk_id}", "available": list(speakers.keys())},
        )

    caption = payload.get("caption", default_caption)

    # Text preprocessing: transliteration → normalization
    raw_text = text
    apply_pron = payload.get("apply_pron", True)
    apply_norm = payload.get("apply_norm", True)
    if apply_pron and pron_transliterator is not None:
        text = pron_transliterator(text)
    if apply_norm:
        text = text_normalizer(text)
    logger.info("Preprocessed [%s]: '%s' → '%s'", spk_id, raw_text, text)

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _produce():
        """Run blocking TTS generator in a thread, push chunks to async queue."""
        try:
            for chunk in model.inference_zero_shot_stream(
                tts_text=text,
                zero_shot_spk_id=spk_id,
                caption=caption,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    async def generate():
        start = time.monotonic()
        total_pcm_bytes = 0
        first_chunk_time = None

        # Start producer in thread pool
        producer = loop.run_in_executor(None, _produce)

        while True:
            chunk = await queue.get()
            if chunk is _SENTINEL:
                break
            # batch=1: chunk is raw bytes
            pcm = chunk if isinstance(chunk, bytes) else chunk.get("pcm_bytes", b"")
            if pcm:
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                total_pcm_bytes += len(pcm)
                yield pcm

        await producer  # ensure thread finished

        elapsed = time.monotonic() - start
        ttfb = (first_chunk_time - start) if first_chunk_time else elapsed
        audio_duration = total_pcm_bytes / (model.sample_rate * 2)
        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        logger.info(
            "Stream TTS [%s]: %.2fs audio, %.2fs processing (TTFB %.3fs), RTF=%.4f",
            spk_id, audio_duration, elapsed, ttfb, rtf,
        )

        trailer = json.dumps({
            "_tts_meta": True,
            "preprocessed_text": text,
            "pron_dic_version": pron_dic_version,
            "audio_duration": round(audio_duration, 3),
            "processing_time": round(elapsed, 3),
            "ttfb": round(ttfb, 3),
            "rtf": round(rtf, 4),
        }).encode("utf-8")
        yield b"\x00\x00\x00\x00" + trailer

    return StreamingResponse(generate(), media_type="audio/pcm")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FastCosyVoice TTS Server")
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml (ignored when GCS_MODEL_PATH is set)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = parser.parse_args()

    # Start full pipeline (GCS copy + model load) in background thread
    init_thread = threading.Thread(target=init_pipeline, args=(args.config,), daemon=True)
    init_thread.start()

    logger.info("Starting server on http://0.0.0.0:%d", args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
