#!/usr/bin/env python
"""FastRTC WebRTC 오디오 스트리밍 예제

오디오 파일을 읽어서 WebRTC를 통해 청크 단위로 스트리밍 재생합니다.
Gradio WebRTC 컴포넌트를 사용하여 구현합니다.
"""

import gradio as gr
import numpy as np
from fastrtc import WebRTC
from pydub import AudioSegment
import os
import json
import time

# 오디오 파일 경로
audio_path = "/home/longtou.2024/mount/longtou/saved/fast_cosyvoice/oneyoung_ref/oneyoung.wav"

def generate_audio():
    # 예시: 440Hz 사인파 생성 (반드시 (sampling_rate, numpy_array) 형태여야 함)
    print("=" * 60)
    print("[generate_audio] 함수가 호출되었습니다!")
    print("=" * 60)
    import sys
    sys.stdout.flush()  # 즉시 출력 보장
    
    sr = 16000
    t = np.linspace(0, 1, sr)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    # 반드시 튜플로 yield
    print(f"[generate_audio] 오디오 데이터 shape: {audio_data.reshape(1, -1).shape}")
    print(f"[generate_audio] 샘플 레이트: {sr}Hz")
    print(f"[generate_audio] 오디오 데이터 타입: {audio_data.dtype}")
    sys.stdout.flush()
    
    yield (sr, audio_data.reshape(1, -1))
    print("[generate_audio] yield 완료")
    sys.stdout.flush()

with gr.Blocks() as demo:
    # 1. 오디오 재생만 하려면 mode="receive" 권장 (마이크 권한 필요 없음)
    audio = WebRTC(label="Stream", mode="receive", modality="audio")
    
    play_btn = gr.Button("오디오 재생 시작")
    
    # 2. WebRTC 컴포넌트는 stream() 메서드를 사용해야 함
    # 중요: receive 모드에서는 trigger 파라미터가 필수입니다!
    # trigger에 버튼의 click 이벤트를 전달하면 버튼 클릭 시 스트림이 시작됨
    
    print("[설정] audio.stream() 호출 전")
    print(f"[설정] play_btn.click 타입: {type(play_btn.click)}")
    print(f"[설정] play_btn.click 값: {play_btn.click}")
    
    # receive 모드에서는 trigger 파라미터가 필수이므로 반드시 제공해야 함
    # trigger에 버튼의 click 이벤트를 전달
    print("[설정] audio.stream() 호출 시작...")
    print(f"[설정] generate_audio 함수: {generate_audio}")
    print(f"[설정] trigger (play_btn.click): {play_btn.click}")
    
    try:
        result = audio.stream(
            fn=generate_audio,
            inputs=[],
            outputs=[audio],
            trigger=play_btn.click  # 버튼 클릭 시 스트림 시작 (필수 파라미터)
        )
        print(f"[설정] audio.stream() 호출 완료 - 반환값: {result}")
    except Exception as e:
        print(f"[설정] audio.stream() 호출 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    demo.launch(share=False)
