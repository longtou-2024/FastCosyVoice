# requirements.txt
# - google-cloud-texttospeech # for google-chirp
# - gradio
# - fastrtc
# 
# 'gcloud auth application-default login' might be required for google-chirp

import gradio as gr
import numpy as np
import asyncio
import threading
import queue
from google.cloud import texttospeech
from fastrtc import WebRTC
import time
import requests

API_URL = "http://localhost:8000/synthesize"

audio_queue = queue.Queue()
text_queue = queue.Queue()
is_producing = False
sample_rate = 24000
producer_thread = None  # 스레드 참조 저장

# 종료 신호용 sentinel 값
STOP_SENTINEL = object()  # 프로듀서 스레드 종료 신호
STREAM_STOP_SENTINEL = object()  # 스트림 generator 종료 신호


def split_text_into_chunks(text):
    return text.split('\n')


def _convert_pcm_to_numpy(pcm_bytes):
    """PCM bytes를 numpy 배열로 변환하는 헬퍼 함수"""
    audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio_array


def produce_chirp(text_queue, audio_queue: queue.Queue, sample_rate):
    """동기 TTS 합성 함수 - 각 청크를 받자마자 큐에 추가"""
    client = texttospeech.TextToSpeechClient()
    
    streaming_audio_config = texttospeech.StreamingAudioConfig(
        audio_encoding=texttospeech.AudioEncoding.PCM,
        sample_rate_hertz=sample_rate,
    )
    
    streaming_config = texttospeech.StreamingSynthesizeConfig(
        voice=texttospeech.VoiceSelectionParams(
            name="ko-KR-Chirp3-HD-Achernar",
            language_code="ko-KR",
        ),
        streaming_audio_config=streaming_audio_config,
    )
    
    config_request = texttospeech.StreamingSynthesizeRequest(
        streaming_config=streaming_config
    )

    while True:
        try:
            # 타임아웃 없이 블로킹 - sentinel 값으로 종료
            text = text_queue.get(block=True, timeout=None)
            
            # 종료 신호 확인
            if text is STOP_SENTINEL:
                print("[Chirp Producer] 종료 신호 수신, 스레드 종료")
                break
            
            def request_generator():
                yield config_request
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(text=text)
                )

            streaming_responses = client.streaming_synthesize(request_generator())
            chunk_index = 0

            for response in streaming_responses:
                if response.audio_content:
                    audio_bytes = response.audio_content
                    chunk_index += 1

                    samples = len(audio_bytes) // 2
                    duration = samples / sample_rate
                    print(f"[Chirp Producer] 수신 청크 #{chunk_index}: PCM bytes 길이 = {len(audio_bytes)} bytes, 재생 시간 = {duration:.3f} 초")

                    # PCM을 numpy 배열로 변환
                    audio_array = _convert_pcm_to_numpy(audio_bytes)

                    # 청크를 받자마자 바로 큐에 추가 (스트리밍 방식)
                    audio_queue.put(audio_array)
                    print(f"[Chirp Producer] 큐에 추가: numpy 배열 길이 = {len(audio_array)} samples")
            print("[Chirp Producer] 모든 청크 전송 완료")
            
        except Exception as e:
            print(f"[Chirp Producer] 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
            continue
    
    print("[Chirp Producer] 스레드 종료 완료")


def produce_cosy(text_queue, audio_queue: queue.Queue, sample_rate, prompt_idx=0):
    """CosyVoice TTS 합성 함수 - 각 청크를 받자마자 큐에 추가"""
    # CosyVoice API 엔드포인트 설정
    api_url = API_URL
    
    while True:
        try:
            # 타임아웃 없이 블로킹 - sentinel 값으로 종료
            text = text_queue.get(block=True, timeout=None)
            
            # 종료 신호 확인
            if text is STOP_SENTINEL:
                print("[CosyVoice Producer] 종료 신호 수신, 스레드 종료")
                break
            
            if not text.strip():
                continue
            
            # HTTP POST 요청으로 스트리밍 응답 받기
            response = requests.post(
                api_url,
                json={"tts_text": text, "prompt_idx": prompt_idx},
                stream=True
            )
            
            chunk_index = 0
            
            # 스트리밍 응답에서 청크 단위로 처리
            for audio_chunk in response.iter_content(chunk_size=None):
                if audio_chunk:
                    chunk_index += 1
                    
                    # PCM bytes를 numpy 배열로 변환
                    audio_array = _convert_pcm_to_numpy(audio_chunk)
                    samples = len(audio_array)
                    duration = samples / sample_rate
                    print(f"[CosyVoice Producer] 수신 청크 #{chunk_index}: PCM bytes 길이 = {len(audio_chunk)} bytes, 재생 시간 = {duration:.3f} 초")
                    
                    # 청크를 받자마자 바로 큐에 추가 (스트리밍 방식)
                    audio_queue.put(audio_array)
                    print(f"[CosyVoice Producer] 큐에 추가: numpy 배열 길이 = {len(audio_array)} samples")
            
            print("[CosyVoice Producer] 모든 청크 전송 완료")
            
        except Exception as e:
            print(f"[CosyVoice Producer] 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
            continue
    
    print("[CosyVoice Producer] 스레드 종료 완료")




# Gradio 인터페이스 생성
def create_gradio_app():
    with gr.Blocks(title="통합 스트리밍 TTS - Chirp & Cosy") as app:
        gr.Markdown("# AI LAB TTS (스트리밍) 데모")
        #gr.Markdown("텍스트를 입력한 후 합성 버튼을 클릭하면 실시간 스트리밍으로 음성이 생성됩니다.")
        
        with gr.Row():
            model_radio = gr.Radio(
                choices=["chirp", "cosy"],
                value="chirp",
                label="모델 선택",
                info="Chirp: Google Cloud TTS API | Cosy: AI LAB TTS 모델"
            )
        
        with gr.Row(visible=False) as cosy_prompt_row:
            cosy_prompt_radio = gr.Radio(
                choices=[0, 1, 2, 3],
                value=0,
                label="CosyVoice Prompt Index",
                info="CosyVoice 음성 스타일 선택 (0-3)",
                type="value"
            )
        
        # 웹브라우저 리프레시 시 큐 초기화 함수
        def clear_queues_on_load():
            """페이지 로드 시 큐를 비우고 스레드를 종료하는 함수"""
            global audio_queue, text_queue, is_producing, producer_thread, STOP_SENTINEL, STREAM_STOP_SENTINEL
            
            # 스레드가 실행 중이면 종료 신호 전송
            if producer_thread and producer_thread.is_alive():
                print("[Queue Clear] 프로듀서 스레드에 종료 신호 전송...")
                # 큐에 종료 신호 넣기 (블로킹된 get()을 깨움)
                text_queue.put(STOP_SENTINEL)
                
                # 스레드 종료 대기
                producer_thread.join(timeout=2.0)
                
                if producer_thread.is_alive():
                    print("[Queue Clear] 경고: 스레드가 2초 내에 종료되지 않았습니다")
                else:
                    print("[Queue Clear] 프로듀서 스레드 종료 완료")
            
            # 스트림 generator 종료를 위한 신호 전송
            print("[Queue Clear] 스트림 generator에 종료 신호 전송...")
            audio_queue.put(STREAM_STOP_SENTINEL)
            
            # 큐 비우기 (종료 신호 제외)
            cleared_audio = 0
            while not audio_queue.empty():
                try:
                    item = audio_queue.get_nowait()
                    # STREAM_STOP_SENTINEL이 아닌 경우만 카운트
                    if item is not STREAM_STOP_SENTINEL:
                        cleared_audio += 1
                except queue.Empty:
                    break
            
            cleared_text = 0
            while not text_queue.empty():
                try:
                    item = text_queue.get_nowait()
                    # STOP_SENTINEL이 아닌 경우만 카운트
                    if item is not STOP_SENTINEL:
                        cleared_text += 1
                except queue.Empty:
                    break
            
            # 상태 초기화
            is_producing = False
            producer_thread = None
            
            print(f"[Queue Clear] 웹브라우저 리프레시 감지: audio_queue {cleared_audio}개, text_queue {cleared_text}개 초기화 완료")
            return None
        
        # 앱 로드 시 큐 초기화 (웹브라우저 리프레시 감지)
        app.load(
            fn=clear_queues_on_load,
            inputs=[],
            outputs=[]
        )
        
        with gr.Row():
            text_tts = gr.Textbox(label="TTS processing...",
                                    lines=5)
        with gr.Row():
            queue_btn = gr.Button("LLM -> TTS", variant="primary")
        with gr.Row():
            text_input = gr.Textbox(
                label="LLM generating...",
                #placeholder="여기에 텍스트를 입력하세요...",
                lines=5,
                value="\n".join([
                    "반가워요. 당신의 귀를 달콤하게 채워줄 에이아이 디제이 허니듀입니다. 오늘은 어떤 노래가 듣고 싶으신가요?",
                    "지금 기분이 어떠신가요? 단어 하나만 말씀해 주시면 제가 찰떡같은 선곡을 준비해 드릴게요.",
                    "오후 세 시, 슬슬 졸음이 밀려오는 시간이죠? 텐션을 확 올릴 수 있는 신나는 댄스곡들을 가져왔습니다.",
                    "창밖에 비가 내리네요. 이런 날엔 차분한 째즈 한 잔 어떠세요? 추천 리스트 일 번부터 삼 번까지 확인해 보세요.",
                    "준비한 세 곡 중에서 가장 마음에 드는 노래를 골라주세요. 당신의 선택이 정말 궁금해지네요.",
                    "이 곡은 멜론 차트에서 일 위를 기록했던 아주 유명한 노래예요. 첫 소절부터 귀에 꽂히실 겁니다.",
                    "오늘 하루도 정말 고생 많으셨어요. 밤 열한 시에 딱 어울리는 포근한 로우파이 음악들을 들려드릴게요.",
                    "운동할 때 듣기 좋은 비트감 있는 음악들을 모아봤어요. 심박수를 올릴 준비 되셨나요?",
                    "방금 들으신 노래가 마음에 들지 않으셨다면, 다른 느낌의 추천 곡도 준비되어 있으니 언제든 말씀해 주세요.",
                    "허니듀와 함께하는 음악 시간, 즐거우셨나요? 내일 이 시간에 더 좋은 노래로 다시 만나요.",
                ])
            )
        
        with gr.Row():
            stream_btn = gr.Button("오디오 스트림 생성", variant="stop")
        
        with gr.Row():
            audio_output = WebRTC(
                label="합성된 음성",
                mode="receive",  # receive 모드: 오디오 재생만 (녹음 버튼 없음)
                modality="audio",
            )
        
        # LLM -> TTS 큐 이동 함수
        def move_first_line_to_queue(text_input_val, text_tts_val):
            """text_input의 첫 번째 줄을 text_tts로 이동 (기존 내용 유지하며 append)"""
            # text_input이 비어있으면 변경 없음
            if not text_input_val or not text_input_val.strip():
                return text_input_val, text_tts_val
            
            lines = text_input_val.split('\n')
            if len(lines) == 0:
                return text_input_val, text_tts_val
            
            # 첫 번째 줄 추출
            first_line = lines[0].strip()
            if not first_line:
                return text_input_val, text_tts_val
            
            # 나머지 줄들
            remaining_lines = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            # text_tts에 기존 내용이 있으면 새 줄 추가, 없으면 새로 생성
            if text_tts_val and text_tts_val.strip():
                updated_queue = text_tts_val + '\n' + first_line
            else:
                updated_queue = first_line
            
            # 나머지를 text_input으로, 업데이트된 큐를 text_tts로 반환
            return remaining_lines, updated_queue
        
        # queue_btn 클릭 이벤트 설정
        queue_btn.click(
            fn=move_first_line_to_queue,
            inputs=[text_input, text_tts],
            outputs=[text_input, text_tts]
        )
        
        # text_tts 변경 감지 및 오디오 청크 저장 함수 (비동기)
        async def process_new_text_and_save_chunks(text_tts_val, model_type, prompt_idx):
            """text_tts에 새 문장이 추가되면 비동기적으로 TTS 합성 후 오디오 청크를 큐에 추가"""
            if not text_tts_val or not text_tts_val.strip():
                return

            global audio_queue, text_queue

            # text_tts의 마지막 줄(새로 추가된 줄) 추출
            lines = text_tts_val.strip().split('\n')
            if len(lines) == 0:
                return

            # 마지막 줄이 새로 추가된 문장
            new_text = lines[-1].strip()
            if not new_text:
                return

            print(f"[TTS Queue] 새 문장 감지 ({model_type}, prompt_idx={prompt_idx}): {new_text[:50]}...")

            text_queue.put(new_text)

            global is_producing, producer_thread
            if not is_producing:
                # 모델 타입에 따라 적절한 Producer 함수 선택
                if model_type == "chirp":
                    producer_func = produce_chirp
                    producer_args = (text_queue, audio_queue, sample_rate)
                else:  # cosy
                    producer_func = produce_cosy
                    producer_args = (text_queue, audio_queue, sample_rate, prompt_idx)
                
                producer_thread = threading.Thread(
                    target=producer_func,
                    args=producer_args,
                    daemon=True
                )
                producer_thread.start()
                is_producing = True
            #_producer_chirp_async(new_text, audio_queue, sample_rate)
        
        # 모델 변경 시 CosyVoice Prompt Index 라디오 버튼 표시/숨김 처리
        def update_prompt_visibility(model_type):
            """모델 선택에 따라 CosyVoice Prompt Index 라디오 버튼 표시/숨김"""
            return gr.update(visible=(model_type == "cosy"))
        
        # 모델 변경 시 기존 프로듀서 스레드 종료 함수
        def handle_model_change(model_type):
            """모델이 변경되면 기존 프로듀서 스레드를 종료하고 새 모델로 재시작 준비"""
            global is_producing, producer_thread
            
            if producer_thread and producer_thread.is_alive():
                print(f"[Model Change] 모델 변경 감지 ({model_type}), 기존 프로듀서 스레드 종료 중...")
                # 큐에 종료 신호 넣기 (블로킹된 get()을 깨움)
                text_queue.put(STOP_SENTINEL)
                
                # 스레드 종료 대기
                producer_thread.join(timeout=2.0)
                
                if producer_thread.is_alive():
                    print("[Model Change] 경고: 스레드가 2초 내에 종료되지 않았습니다")
                else:
                    print("[Model Change] 프로듀서 스레드 종료 완료")
            
            # 상태 초기화
            is_producing = False
            producer_thread = None
            print(f"[Model Change] 새 모델로 전환 준비 완료: {model_type}")
        
        # 모델 변경 감지 이벤트 설정 (프로듀서 스레드 종료 및 UI 업데이트)
        model_radio.change(
            fn=handle_model_change,
            inputs=[model_radio],
            outputs=[]
        )
        
        # 모델 변경 시 CosyVoice Prompt Index 라디오 버튼 표시/숨김
        model_radio.change(
            fn=update_prompt_visibility,
            inputs=[model_radio],
            outputs=[cosy_prompt_row]
        )
        
        # text_tts 변경 감지 이벤트 설정
        text_tts.change(
            fn=process_new_text_and_save_chunks,
            inputs=[text_tts, model_radio, cosy_prompt_radio],
            outputs=[]
        )
        
        # 저장된 오디오 청크를 스트림으로 재생하는 함수 (일반 generator, 비동기 폴링)
        def create_stream():
            """스트림을 열고 오디오 청크 큐를 반복적으로 확인하여 오디오 청크를 반환"""
            global audio_queue, sample_rate, STREAM_STOP_SENTINEL
            chunk_index = 0
            
            print(f"[Stream] 스트림 시작")
            t_before_yield = time.time()
            prev_duration = -1
            # 비동기 폴링: 큐에 입력이 들어오길 대기하고 반복적으로 확인
            while True:
                try:
                    # 타임아웃 없이 블로킹 - sentinel 값으로 종료
                    audio_chunk = audio_queue.get(block=True, timeout=None)
                    
                    # 종료 신호 확인
                    if audio_chunk is STREAM_STOP_SENTINEL:
                        print("[Stream] 종료 신호 수신, 스트림 종료")
                        break
                    
                    chunk_index += 1
                    duration = len(audio_chunk) / sample_rate
                    print(f"[Stream] 청크 #{chunk_index} 재생: {len(audio_chunk)} samples (약 {duration:.3f} 초)")

                    elapsed_time = time.time() - t_before_yield
                    if elapsed_time < prev_duration:
                        time.sleep(prev_duration-elapsed_time)
                    
                    t_before_yield = time.time()
                    prev_duration = duration
                    yield (sample_rate, audio_chunk)
                    #time.sleep(duration)
                        
                except Exception as e:
                    print(f"[Stream] 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.01)
                    continue
            
            print("[Stream] 스트림 종료 완료")

        # stream 이벤트 설정 - receive 모드에서는 trigger 파라미터가 필수
        # 저장된 오디오 청크를 재생하도록 변경
        audio_output.stream(
            fn=create_stream,
            inputs=[],
            outputs=audio_output,
            trigger=stream_btn.click,
            time_limit=300  # 60s
        )
    
    app.queue()
    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
