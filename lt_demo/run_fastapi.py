# uvicorn lt_demo.main:app --host 0.0.0.0 --port 8000
import os
os.environ["TRITON_LOG_LEVEL"] = "ERROR"
import sys
sys.path.append('third_party/Matcha-TTS')

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import torch
import asyncio
from queue import Queue
from threading import Thread
import logging

from lt_demo_stream_api import load_model, synthesize_streaming
from lt_demo_stream_api import load_model_basic, synthesize_streaming_basic
from lt_demo_stream_api import synthesize_streaming_kayden

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# 전역 모델 변수
model = None

def _convert_pcm_to_numpy(pcm_bytes):
    """PCM bytes를 numpy 배열로 변환하는 헬퍼 함수"""
    audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio_array

@app.on_event("startup")
async def startup_event():
    global model
    model, prompt_text, spk_id, cosy_sample_rate = load_model()

class SynthesizeRequest(BaseModel):
    tts_text: str
    prompt_idx: int = 0

async def generate_audio_stream(request: SynthesizeRequest):
    """오디오 청크를 스트리밍하는 비동기 제너레이터 - 생성되는 대로 바로 반환"""
    kwargs = {
        "text": request.tts_text,
        "prompt_text": "장원영의 칠초 인터뷰 시작하겠습니다~",
        "spk_id": '',
        "sample_rate": 24000,
    }
    
    # 큐를 사용하여 별도 스레드에서 생성된 청크를 받기
    queue = Queue()
    exception = None
    
    def run_synthesis():
        """별도 스레드에서 실행되는 함수"""
        nonlocal exception
        chunk_count = 0
        try:
            logger.info(f"[큐 입력] 합성 시작 - 텍스트: {kwargs['text']}")
            for chunk in synthesize_streaming_kayden(model, **kwargs):
                # tensor인 경우 numpy 배열로 변환
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()
                    # numpy 배열을 int16으로 변환
                    audio_int16 = (chunk * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                elif isinstance(chunk, np.ndarray):
                    #chunk = np.array(chunk)
                    # numpy 배열을 int16으로 변환
                    audio_int16 = (chunk * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                else:
                    audio_bytes = chunk

                chunk_count += 1
                queue.put(audio_bytes)
                logger.info(f"[큐 입력] 청크 #{chunk_count} 큐에 추가됨 (크기: {len(audio_bytes)} bytes)")
            queue.put(None)  # 완료 신호
            logger.info(f"[큐 입력] 합성 완료 - 총 {chunk_count}개 청크 생성됨")
        except Exception as e:
            logger.error(f"[큐 입력] 오류 발생: {e}", exc_info=True)
            exception = e
            queue.put(None)
    
    # 별도 스레드에서 합성 시작
    thread = Thread(target=run_synthesis, daemon=True)
    thread.start()
    
    # 큐에서 청크를 비동기로 받아서 반환
    chunk_count = 0
    logger.info("[큐 출력] 청크 수신 대기 시작")
    while True:
        # 큐에서 가져오기 (비동기적으로 대기)
        chunk = await asyncio.to_thread(queue.get)
        if chunk is None:
            logger.info(f"[큐 출력] 모든 청크 수신 완료 - 총 {chunk_count}개 청크 반환됨")
            break
        if exception:
            logger.error(f"[큐 출력] 예외 발생: {exception}")
            raise exception
        chunk_count += 1
        logger.info(f"[큐 출력] 청크 #{chunk_count} 큐에서 꺼냄 (크기: {len(chunk)} bytes) - 클라이언트로 전송")
        yield chunk

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """스트리밍 음성합성 엔드포인트 - 청크가 생성되는 대로 바로 반환
    
    반환 형식: Raw PCM 오디오 (16-bit, 24kHz, 모노)
    클라이언트는 이를 받아서 바로 재생하거나 WAV 파일로 저장할 수 있습니다.
    """
    return StreamingResponse(
        generate_audio_stream(request),
        media_type="audio/pcm",
        headers={
            "Content-Type": "audio/pcm; rate=24000; channels=1; bit=16",
            "Transfer-Encoding": "chunked"
        }
    )

@app.get("/")
async def root():
    return {"message": "CosyVoice TTS Streaming API"}
