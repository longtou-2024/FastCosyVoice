## on GCP VM (L4 gpu instance)
```
git clone -b lt-inference https://github.com/longtou-2024/FastCosyVoice.git
cd FastCosyVoice/
# 버킷으로부터 데이터 복사
gcloud storage cp -r gs://ai-lab-speech-bucket/longtou/share/20260309/data .
uv sync
./run_fastapi.sh # fastapi 서버 실행; 음성합성 모델 서빙
```
from apis import load_model 로 모델 로딩하고  
from apis import synthesize_streaming 로 음성합성하는 코드입니다.


## on local mac
```
# 데모를 위해서 아래 파일을 로컬 맥북에서 실행하였습니다.
# (gradio 앱 데모인데 오디오 컴포넌트가 스트리밍 재생이 끊기는 이슈가 있어서 fastrtc 모듈을 사용했습니다.)
python run_gradio_fastrtc.py
```
