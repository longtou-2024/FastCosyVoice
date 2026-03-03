import gradio as gr
import numpy as np
import time
from fastrtc import WebRTC

def dummy_tts_handler(text: str):
    """
    TTS API 대신 임의의 사인파(Sine Wave)를 생성하여 
    끊김 없는 스트리밍을 구현하는 제너레이터 함수입니다.
    """
    if not text or not text.strip():
        return

    # 오디오 설정
    sample_rate = 24000
    chunk_duration = 0.1  # 0.1초 단위로 청크 생성 (100ms)
    total_duration = 2.0  # 총 2초 동안 재생
    num_samples_per_chunk = int(sample_rate * chunk_duration)
    
    print(f"입력된 텍스트: {text} - 스트리밍 시작...")

    # 0.1초씩 데이터를 끊어서 생성 및 yield
    for i in range(int(total_duration / chunk_duration)):
        # 현재 시간 기반으로 위상(Phase)이 이어지도록 사인파 생성
        start_time = i * chunk_duration
        t = np.linspace(start_time, start_time + chunk_duration, num_samples_per_chunk, endpoint=False)
        
        # 주파수가 조금씩 변하게 설정 (440Hz -> 880Hz)
        frequency = 440 + (i * 40)
        audio_chunk = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        # 실제 API 연산 시간을 시뮬레이션하기 위한 짧은 대기
        time.sleep(0.05) 
        
        # WebRTC 형식: (sample_rate, numpy_array) where array is 2D (1, samples)
        yield (sample_rate, audio_chunk.reshape(1, -1))
        
    print("스트리밍 완료")

# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    gr.Markdown("### FastRTC 실시간 오디오 스트리밍 테스트 (SSH 터널링 대응)")
    
    with gr.Row():
        input_text = gr.Textbox(label="텍스트 입력", placeholder="아무 글자나 입력하고 엔터를 누르세요...")
        # WebRTC 컴포넌트: mode="receive"로 설정하여 서버에서 클라이언트로 오디오 전송
        output_audio = WebRTC(label="실시간 스트리밍 오디오", mode="receive", modality="audio")

    # 텍스트 입력 시 스트리밍 핸들러 실행
    # WebRTC 컴포넌트의 stream 메서드를 사용해야 합니다.
    output_audio.stream(
        fn=dummy_tts_handler, 
        inputs=[input_text], 
        outputs=[output_audio],
        trigger=input_text.submit
    )

if __name__ == "__main__":
    # SSH 터널링 포트 포워딩을 위해 0.0.0.0으로 실행
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
