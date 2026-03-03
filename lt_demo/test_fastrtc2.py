rm /home/longtou.2024/projects/FastCosyVoice/.cursor/debug.logimport gradio as gr
from fastrtc import WebRTC
import numpy as np

def generate_audio():
    # 예시: 440Hz 사인파 생성 (반드시 (sampling_rate, numpy_array) 형태여야 함)
    sr = 16000
    t = np.linspace(0, 1, sr)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    # 반드시 튜플로 yield
    yield (sr, audio_data.reshape(1, -1))

with gr.Blocks() as demo:
    # 1. 오디오 재생만 하려면 mode="receive" 권장 (마이크 권한 필요 없음)
    audio = WebRTC(label="Stream", mode="receive", modality="audio")
    
    play_btn = gr.Button("오디오 재생 시작")
    
    # 2. trigger 속성에 버튼의 click 이벤트를 연결해야 함
    audio.stream(
        fn=generate_audio,
        inputs=[],
        outputs=[audio],
        trigger=play_btn.click  # 이 부분이 누락되면 버튼을 눌러도 반응이 없습니다.
    )

demo.launch()
