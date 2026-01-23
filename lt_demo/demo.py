#!/usr/bin/env python
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Google Cloud Text-To-Speech API streaming sample application with Gradio UI.

Example usage:
    python demo.py
"""

import gradio as gr
import numpy as np
import wave
import io
import asyncio
import queue
import threading
#from google.cloud import texttospeech

from lt_demo_stream_api import load_model, synthesize_streaming

# global
cosyvoice = None
prompt_text = None
spk_id = None
sample_rate = None

def split_text_into_chunks(text):
    return text.split('\n')


def _convert_pcm_to_numpy(pcm_bytes):
    """PCM bytes를 numpy 배열로 변환하는 헬퍼 함수"""
    # 16-bit PCM을 numpy 배열로 변환
    audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio_array


def _producer(text, audio_queue, sample_rate, stop_event):
    """Producer: 음성합성 청크를 받아서 큐에 추가하는 백그라운드 작업"""

    text_chunks = split_text_into_chunks(text)

    try:
        for text_chunk in text_chunks:
            chunk_index = 0
            for audio_bytes in synthesize_streaming(cosyvoice, text_chunk, prompt_text, spk_id, sample_rate):

                if stop_event.is_set():
                    break

                chunk_index += 1

                samples = len(audio_bytes) // 2
                duration = samples / sample_rate
                print(f"[Producer] 수신 청크 #{chunk_index}: PCM bytes 길이 = {len(audio_bytes)} bytes, 재생 시간 = {duration:.3f} 초")

                # PCM을 numpy 배열로 변환하여 큐에 추가 (속도 개선)
                audio_array = _convert_pcm_to_numpy(audio_bytes)
                audio_queue.put(audio_array)
                print(f"[Producer] 큐에 추가: numpy 배열 길이 = {len(audio_array)} samples")

        # 완료 신호
        audio_queue.put(None)
        print("[Producer] 모든 청크 전송 완료")

    except Exception as e:
        print(f"[Producer] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        audio_queue.put(None)  # 에러 발생 시에도 완료 신호


async def _consumer(audio_queue, stop_event):
    """Consumer: 큐에서 오디오 청크를 가져와서 yield하는 비동기 작업
    
    약 2초 분량의 청크가 모일 때마다 yield합니다.
    """
    sample_rate = 24000  # Hz
    target_samples = sample_rate * 2 * 3  # 약 2초 분량 (48000 samples)
    
    # 버퍼링을 위한 변수
    buffer_chunks = []  # numpy 배열들을 저장할 버퍼
    accumulated_samples = 0  # 누적된 샘플 수
    chunk_index = 0
    
    while True:
        if stop_event.is_set():
            break
        
        try:
            # 큐에서 데이터 가져오기 (타임아웃 설정)
            try:
                audio_array = audio_queue.get(timeout=0.1)
            except queue.Empty:
                # 큐가 비어있으면 잠시 대기 후 다시 시도
                await asyncio.sleep(0.01)
                continue
            
            # 완료 신호 확인
            if audio_array is None:
                # 마지막 남은 버퍼가 있으면 yield
                if buffer_chunks:
                    chunk_index += 1
                    combined_array = np.concatenate(buffer_chunks)
                    duration = len(combined_array) / sample_rate
                    print(f"[Consumer] 마지막 버퍼링된 청크 #{chunk_index} yield: {len(combined_array)} samples (약 {duration:.3f} 초)")
                    yield (sample_rate, combined_array)
                
                print("[Consumer] 모든 청크 처리 완료")
                break
            
            # numpy 배열을 버퍼에 추가
            buffer_chunks.append(audio_array)
            accumulated_samples += len(audio_array)
            
            # 약 2초 분량이 모이면 yield
            if accumulated_samples >= target_samples:
                chunk_index += 1
                combined_array = np.concatenate(buffer_chunks)
                duration = len(combined_array) / sample_rate
                print(f"[Consumer] 버퍼링된 청크 #{chunk_index} yield: {len(combined_array)} samples (약 {duration:.3f} 초)")
                yield (sample_rate, combined_array)
                
                # 버퍼 초기화
                buffer_chunks = []
                accumulated_samples = 0
            
            # 다른 작업이 실행될 수 있도록 양보
            await asyncio.sleep(0)
            
        except Exception as e:
            print(f"[Consumer] 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            break




async def synthesize_and_play(text, state):
    """텍스트를 합성하고 오디오 청크를 스트리밍으로 반환합니다.
    
    Producer-Consumer 패턴으로 구현:
    - Producer: 백그라운드 스레드에서 음성합성 청크를 받아 큐에 추가
    - Consumer: 비동기로 큐에서 청크를 가져와 yield
    """
    if not text or not text.strip():
        return
    
    sample_rate = 24000  # Hz
    
    # 이전 작업이 있으면 중지하고 완료될 때까지 대기
    if state is not None:
        if 'stop_event' in state:
            state['stop_event'].set()
        
        # 이전 Producer 스레드가 완료될 때까지 대기
        if 'producer_thread' in state and state['producer_thread'] is not None:
            print("[Main] 이전 Producer 스레드 완료 대기 중...")
            state['producer_thread'].join(timeout=2.0)  # 최대 2초 대기
            print("[Main] 이전 Producer 스레드 완료")
    
    # 새로운 상태 초기화
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    state = {
        'queue': audio_queue,
        'stop_event': stop_event,
        'producer_thread': None
    }
    
    # Producer 스레드 시작
    producer_thread = threading.Thread(
        target=_producer,
        args=(text, audio_queue, sample_rate, stop_event),
        daemon=True
    )
    producer_thread.start()
    state['producer_thread'] = producer_thread
    
    print("[Main] Producer 스레드 시작, Consumer 시작")
    
    # Consumer 실행 (비동기)
    # Gradio outputs에 [audio_output, state]가 있으므로 각 yield마다 튜플 반환
    async for audio_data in _consumer(audio_queue, stop_event):
        yield audio_data, state
    
    print("[Main] 모든 작업 완료")
    # 마지막에 state도 반환
    yield None, state


# Gradio 인터페이스 생성
def create_gradio_app():
    with gr.Blocks(title="Google Chirp Streaming TTS") as app:
        gr.Markdown("# Google Chirp Streaming TTS")
        gr.Markdown("텍스트를 입력하고 합성 버튼을 클릭하면 실시간 스트리밍으로 음성이 생성됩니다.")
        
        # State로 Producer-Consumer 상태 관리
        state = gr.State(value=None)
        
        with gr.Row():
            text_input = gr.Textbox(
                label="합성할 텍스트 입력",
                placeholder="여기에 텍스트를 입력하세요...",
                lines=5,
                value="안녕하세요 카카오 엔터테인먼트 크루 여러분?\n만나서 정말 반갑습니다.\n오늘 날씨가 매우 추우니 출근길 조심하셔요~"
            )
        
        with gr.Row():
            synthesize_btn = gr.Button("음성 합성 시작", variant="primary")
        
        with gr.Row():
            audio_output = gr.Audio(
                label="합성된 음성",
                type="numpy",
                streaming=True,
                autoplay=True,
                format="wav",
            )
        
        synthesize_btn.click(
            fn=synthesize_and_play,
            inputs=[text_input, state],
            outputs=[audio_output, state]
        )
    
    app.queue()
    return app


if __name__ == "__main__":
    cosyvoice, prompt_text, spk_id, sample_rate = load_model()
    app = create_gradio_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
