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
from google.cloud import texttospeech



def split_text_into_chunks(text):
    return text.split('\n')


def _convert_pcm_to_wav(pcm_bytes, sample_rate):
    """PCM bytes를 WAV 포맷으로 변환하는 헬퍼 함수"""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 모노
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_buffer.getvalue()


def _producer(text, audio_queue, sample_rate, stop_event):
    """Producer: 음성합성 청크를 받아서 큐에 추가하는 백그라운드 작업"""
    try:
        client = texttospeech.TextToSpeechClient()
        
        streaming_audio_config = texttospeech.StreamingAudioConfig(
            audio_encoding=texttospeech.AudioEncoding.PCM,
            sample_rate_hertz=sample_rate,
        )
        
        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(
                name="en-US-Chirp3-HD-Charon",
                language_code="en-US",
            ),
            streaming_audio_config=streaming_audio_config,
        )
        
        config_request = texttospeech.StreamingSynthesizeRequest(
            streaming_config=streaming_config
        )
        
        text_chunks = split_text_into_chunks(text)
        
        def request_generator():
            yield config_request
            for chunk in text_chunks:
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(text=chunk)
                )
        
        streaming_responses = client.streaming_synthesize(request_generator())
        chunk_index = 0
        
        for response in streaming_responses:
            if stop_event.is_set():
                break
                
            if response.audio_content:
                audio_bytes = response.audio_content
                chunk_index += 1
                
                samples = len(audio_bytes) // 2
                duration = samples / sample_rate
                print(f"[Producer] 수신 청크 #{chunk_index}: PCM bytes 길이 = {len(audio_bytes)} bytes, 재생 시간 = {duration:.3f} 초")
                
                # PCM을 WAV로 변환하여 큐에 추가
                wav_bytes = _convert_pcm_to_wav(audio_bytes, sample_rate)
                audio_queue.put(wav_bytes)
                print(f"[Producer] 큐에 추가: WAV bytes 길이 = {len(wav_bytes)} bytes")
        
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
    target_pcm_bytes = sample_rate * 2 * 2  # 약 2초 분량 (96000 bytes)
    
    # 버퍼링을 위한 변수
    buffer_chunks = []  # WAV 청크들을 저장할 버퍼
    accumulated_pcm_bytes = 0  # 누적된 PCM 바이트 수
    chunk_index = 0
    
    while True:
        if stop_event.is_set():
            break
        
        try:
            # 큐에서 데이터 가져오기 (타임아웃 설정)
            try:
                wav_bytes = audio_queue.get(timeout=0.1)
            except queue.Empty:
                # 큐가 비어있으면 잠시 대기 후 다시 시도
                await asyncio.sleep(0.01)
                continue
            
            # 완료 신호 확인
            if wav_bytes is None:
                # 마지막 남은 버퍼가 있으면 yield
                if buffer_chunks:
                    chunk_index += 1
                    combined_wav = _combine_wav_chunks(buffer_chunks, sample_rate)
                    print(f"[Consumer] 마지막 버퍼링된 청크 #{chunk_index} yield: {len(combined_wav)} bytes (약 {accumulated_pcm_bytes / sample_rate / 2:.3f} 초)")
                    yield combined_wav
                
                print("[Consumer] 모든 청크 처리 완료")
                break
            
            # WAV 파일에서 PCM 데이터 길이 추정 (WAV 헤더 44 bytes 제외)
            estimated_pcm_bytes = len(wav_bytes) - 44
            buffer_chunks.append(wav_bytes)
            accumulated_pcm_bytes += estimated_pcm_bytes
            
            # 약 1초 분량이 모이면 yield
            if accumulated_pcm_bytes >= target_pcm_bytes:
                chunk_index += 1
                combined_wav = _combine_wav_chunks(buffer_chunks, sample_rate)
                print(f"[Consumer] 버퍼링된 청크 #{chunk_index} yield: {len(combined_wav)} bytes (약 {accumulated_pcm_bytes / sample_rate / 2:.3f} 초)")
                yield combined_wav
                
                # 버퍼 초기화
                buffer_chunks = []
                accumulated_pcm_bytes = 0
            
            # 다른 작업이 실행될 수 있도록 양보
            await asyncio.sleep(0)
            
        except Exception as e:
            print(f"[Consumer] 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            break


def _combine_wav_chunks(wav_chunks, sample_rate):
    """여러 WAV 청크를 하나의 WAV 파일로 합치는 헬퍼 함수"""
    if not wav_chunks:
        return None
    
    if len(wav_chunks) == 1:
        return wav_chunks[0]
    
    # 모든 WAV 파일의 PCM 데이터를 추출하여 합치기
    all_pcm_data = bytearray()
    
    for wav_bytes in wav_chunks:
        # WAV 파일 파싱하여 PCM 데이터 추출
        wav_buffer = io.BytesIO(wav_bytes)
        with wave.open(wav_buffer, 'rb') as wav_file:
            pcm_data = wav_file.readframes(wav_file.getnframes())
            all_pcm_data.extend(pcm_data)
    
    # 합쳐진 PCM 데이터를 하나의 WAV 파일로 변환
    combined_wav_buffer = io.BytesIO()
    with wave.open(combined_wav_buffer, 'wb') as combined_wav:
        combined_wav.setnchannels(1)  # 모노
        combined_wav.setsampwidth(2)  # 16-bit = 2 bytes
        combined_wav.setframerate(sample_rate)
        combined_wav.writeframes(all_pcm_data)
    
    return combined_wav_buffer.getvalue()


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
    async for wav_bytes in _consumer(audio_queue, stop_event):
        yield wav_bytes, state
    
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
                value="Hello there.\nHow are you today?\nIt's such nice weather outside."
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
    
    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
