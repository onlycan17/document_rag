# 🌞 업스테이지 Solar 및 임베딩 설정 가이드

이 프로젝트는 업스테이지(Upstage)의 Solar LLM과 고성능 한국어 임베딩 모델을 지원합니다. 이 가이드는 업스테이지 관련 설정을 안내합니다.

## 1. API 키 설정

먼저, 업스테이지에서 발급받은 API 키를 `.env` 파일에 추가해야 합니다.

```env
# .env 파일
UPSTAGE_API_KEY="your-upstage-api-key-here"
```

API 키를 설정하면, 앱 사이드바의 "LLM 모델 설정"에서 Upstage Solar 모델을 선택하여 사용할 수 있습니다.

## 2. 업스테이지 임베딩 설치 (선택 사항)

이 프로젝트의 기본 임베딩 모델 외에 업스테이지의 임베딩 모델을 사용하고 싶을 경우, 별도의 설치가 필요합니다.

**⚠️ 중요**: `langchain-upstage` 패키지는 `tokenizers<0.21` 버전을 강제하여, 최신 `transformers` 라이브러리와 버전 충돌을 일으킬 수 있습니다. 이 문제를 피하기 위해 `--no-deps` 플래그를 사용하여 의존성 없이 설치하는 것을 권장합니다.

```bash
# 가상환경이 활성화된 상태에서 실행
pip install --no-deps langchain-upstage
```

설치 후, `.env` 파일에 `UPSTAGE_API_KEY`가 설정되어 있다면, 문서 처리 시 업스테이지 임베딩 모델이 자동으로 사용될 수 있습니다. (단, `config.py`의 `embedding_model_name` 설정에 따라 동작이 달라질 수 있습니다.)

만약 버전 충돌 문제가 발생하면, `langchain-upstage`를 삭제하고 다시 `pip install -r requirements.txt`를 실행하여 기본 환경으로 복구할 수 있습니다.