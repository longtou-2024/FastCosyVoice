## on GCP VM (L4 gpu instance)
```
git clone -b lt-inference https://github.com/longtou-2024/FastCosyVoice.git
cd FastCosyVoice/
# 버킷으로부터 데이터 복사
gcloud storage cp -r gs://ai-lab-speech-bucket/longtou/share/20260330/data .
./install.sh
uv run python run_mp_tts.py # 배치 스트리밍 추론 예시 코드
>>>
# output/mp_tts_test_multi_v3/ 아래 wav 파일이 생성됩니다.
```
