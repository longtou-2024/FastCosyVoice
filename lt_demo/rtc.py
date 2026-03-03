#!/usr/bin/env python
"""
FastRTC WebRTC 오디오 스트리밍 데모

합성 버튼을 클릭하면 미리 정의된 오디오가 WebRTC를 통해 실시간으로 스트리밍 재생됩니다.
FastRTC 문서: https://fastrtc.org/userguide/gradio/
"""

import gradio as gr
import numpy as np
from fastrtc import WebRTC
import sys
import os

# 샘플 레이트
SAMPLE_RATE = 16000

# WebRTC ICE 서버 설정
# SSH 터널링 환경에서 UDP 패킷 통신을 위해 STUN/TURN 서버 설정 필요
rtc_configuration = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"}
    ]
}

# TURN 서버가 있으면 추가 (SSH 터널링 환경에서 UDP 통신이 어려울 경우 필요)
turn_url = os.environ.get("TURN_URL")
turn_username = os.environ.get("TURN_USERNAME")
turn_password = os.environ.get("TURN_PASSWORD")

if turn_url and turn_username and turn_password:
    rtc_configuration["iceServers"].append({
        "urls": turn_url,
        "username": turn_username,
        "credential": turn_password,
    })
    print(f"[RTC 설정] TURN 서버 추가됨: {turn_url}")
else:
    print("[RTC 설정] STUN 서버만 사용 (SSH 터널링 환경에서는 TURN 서버 권장)")

def generate_audio():
    """
    미리 정의된 오디오를 생성하여 WebRTC로 스트리밍합니다.
    
    Yields:
        (sample_rate, numpy_array) 형태의 오디오 튜플
    """
    print("=" * 60)
    print("[generate_audio] 오디오 생성 시작")
    print("=" * 60)
    sys.stdout.flush()
    
    # 간단한 멜로디 생성 (도레미파솔라시도)
    duration = 3.0  # 3초
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    # 각 음표의 주파수 (Hz)
    notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # 도레미파솔라시도
    note_duration = duration / len(notes)
    
    audio_data = np.zeros(int(SAMPLE_RATE * duration), dtype=np.int16)
    
    for i, freq in enumerate(notes):
        start_idx = int(i * note_duration * SAMPLE_RATE)
        end_idx = int((i + 1) * note_duration * SAMPLE_RATE)
        note_t = np.linspace(0, note_duration, end_idx - start_idx)
        
        # 사인파 생성 (부드러운 시작/끝을 위한 envelope 적용)
        note_samples = int(SAMPLE_RATE * note_duration)
        envelope = np.ones(note_samples)
        # Fade in/out
        fade_samples = int(0.05 * SAMPLE_RATE)  # 50ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        note_audio = np.sin(2 * np.pi * freq * note_t) * envelope
        audio_data[start_idx:end_idx] = (note_audio * 32767).astype(np.int16)
    
    # WebRTC 형식: (sample_rate, numpy_array) where array is 2D (1, samples)
    audio_2d = audio_data.reshape(1, -1)
    
    print(f"[generate_audio] 오디오 생성 완료: {len(audio_data)} samples @ {SAMPLE_RATE}Hz")
    sys.stdout.flush()
    
    yield (SAMPLE_RATE, audio_2d)

# Gradio 앱 생성
with gr.Blocks(title="FastRTC 오디오 스트리밍 데모") as demo:
    gr.HTML(
        """
        <h1 style='text-align: center; margin-bottom: 20px;'>
        🎵 FastRTC WebRTC 오디오 스트리밍 데모 ⚡️
        </h1>
        <p style='text-align: center; color: #666;'>
        합성 버튼을 클릭하면 미리 정의된 멜로디가 실시간으로 스트리밍 재생됩니다.
        </p>
        """
    )
    
    with gr.Row():
        with gr.Column():
            # 재생 버튼
            play_btn = gr.Button(
                "🎵 오디오 재생 시작",
                variant="primary",
                size="lg"
            )
        
        with gr.Column():
            # WebRTC 컴포넌트
            # mode="receive": 오디오 재생만 (마이크 권한 불필요)
            # rtc_configuration: SSH 터널링 환경에서 UDP 패킷 통신을 위한 STUN/TURN 서버 설정
            audio = WebRTC(
                label="음성 스트리밍",
                mode="receive",
                modality="audio",
                rtc_configuration=rtc_configuration,  # STUN/TURN 서버 설정
            )
    
    # WebRTC 스트림 이벤트 설정
    # receive 모드에서는 trigger 파라미터가 필수입니다
    # 버튼 클릭 시 generate_audio 함수가 호출됩니다
    audio.stream(
        fn=generate_audio,
        inputs=[],  # 입력 없음
        outputs=[audio],
        trigger=play_btn.click  # 버튼 클릭 시 스트림 시작
    )

if __name__ == "__main__":
    # Gradio 앱 실행
    # share=True: Gradio 공유 링크 생성 (포트포워딩 불필요)
    print("\n" + "=" * 60)
    print("🚀 FastRTC WebRTC 오디오 스트리밍 데모 시작")
    print("=" * 60)
    print("📝 사용법:")
    print("   '오디오 재생 시작' 버튼을 클릭하세요")
    print("   WebRTC를 통해 실시간으로 멜로디가 재생됩니다")
    print("")
    print("🌐 SSH 터널링 환경:")
    print("   - STUN 서버 설정 완료")
    if not (turn_url and turn_username and turn_password):
        print("   - TURN 서버 미설정 (필요시 환경변수로 설정 가능)")
        print("     export TURN_URL=turn:your.turn.server:3478")
        print("     export TURN_USERNAME=your_username")
        print("     export TURN_PASSWORD=your_password")
    print("=" * 60 + "\n")
    
    demo.launch(
        share=False,  # Gradio 공유 링크 생성 (포트포워딩 불필요)
        server_name="0.0.0.0",
        server_port=7860
    )
