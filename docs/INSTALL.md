# 설치 가이드

이 RAG 챗봇 애플리케이션은 원활한 동작을 위해 Python 패키지와 시스템 패키지(OCR 등) 설치가 필요합니다.

## 빠른 시작

### 1. Python 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 시스템 의존성 설치(OCR 지원)

스캔된 PDF의 OCR 기능을 사용하려면 시스템 패키지 설치가 필요합니다:

```bash
bash scripts/setup/install_ocr.sh
```

해당 스크립트는 다음을 설치합니다:
- **Tesseract OCR**: 광학 문자 인식 엔진
- **Poppler 유틸리티**: PDF → 이미지 변환 도구
- **한국어 언어팩**: 한국어 텍스트 인식 지원

### 대안: Docker 사용(준비 중)

모든 의존성이 사전 설치된 Docker 이미지가 곧 제공될 예정입니다.

## 플랫폼별 안내

### macOS
- Homebrew가 필요합니다(시스템 패키지 설치).
- 제공된 스크립트로 모든 의존성을 설치할 수 있습니다.

### Linux (Ubuntu/Debian)
- 시스템 패키지는 apt-get을 사용합니다.
- sudo 권한이 필요할 수 있습니다.

### Linux (RedHat/CentOS)
- 시스템 패키지는 yum을 사용합니다.
- sudo 권한이 필요할 수 있습니다.

### Windows
- 아직 공식 지원하지 않습니다.
- WSL2(Windows Subsystem for Linux) 사용을 권장합니다.

## 의존성 상세

### Python 패키지(pip)
- **Core**: streamlit, langchain 생태계
- **Vector DB**: faiss-cpu, chromadb
- **ML/AI**: sentence-transformers, torch, transformers
- **PDF 처리**: pypdf, PyPDF2, pdf2image[jpeg]
- **OCR**: pytesseract(Tesseract의 Python 래퍼)
- **API 클라이언트**: openai, anthropic, google-generativeai
 - **외부 라우터**: OpenRouter(HTTP, OpenAI 호환)

### 업스테이지 임베딩 사용 시 추가 설치
- 이 프로젝트는 기본 `requirements.txt`에서 `langchain-upstage`를 제외합니다. 이유: `langchain-upstage`가 `tokenizers<0.21`을 강제하여 최신 `transformers`(tokenizers>=0.21)와 충돌하기 때문입니다.
- 업스테이지 임베딩을 사용할 경우, 다음과 같이 의존성 없이 개별 설치하세요:

```bash
pip install --no-deps langchain-upstage
```

- 설치 후 `.env`에 `UPSTAGE_API_KEY`를 설정하세요.

### 시스템 패키지(패키지 관리자)
- **tesseract-ocr**: OCR 엔진
- **tesseract-ocr-kor**: 한국어 언어팩
- **poppler-utils**: PDF 렌더링(pdf2image가 필요로 함)

## 설치 검증

설치 이후, 다음 명령으로 정상 동작을 확인하세요:

```bash
# Python 의존성 확인
python -c "import streamlit, langchain, pytesseract; print('Python dependencies OK')"

# 시스템 의존성 확인
tesseract --version
pdftoppm -v
```

업스테이지 임베딩 사용 시 추가 확인:

```bash
python -c "import langchain_upstage; print('langchain-upstage OK')"

## OpenRouter 사용 설정

`.env`에 아래 값을 설정하세요(예시):

```
IMAGE_ANALYSIS_PROVIDER=openrouter
OPNEROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_API_BASE=https://openrouter.ai/api
OPENROUTER_MM_MODEL=z-ai/glm-4.5v
```

멀티모달 모델 확장이 필요하면 아래 키로 추가할 수 있습니다(콤마 구분):

```
EXTRA_MULTIMODAL_OPENAI_MODELS=gpt-5-mini,gpt-5-nano
EXTRA_MULTIMODAL_GOOGLE_MODELS=gemini-2.5-pro
EXTRA_MULTIMODAL_ANTHROPIC_MODELS=claude-opus-4-1-20250805
```
```

## 문제 해결

### OCR가 동작하지 않을 때
- Tesseract 설치 확인: `which tesseract`
- 언어팩 확인: `tesseract --list-langs`

### PDF → 이미지 변환 실패
- poppler 설치 확인: `which pdftoppm`
- macOS: `brew install poppler`
- Linux: `sudo apt-get install poppler-utils`

### Import 오류
- 패키지 설치 확인: `pip install -r requirements.txt`
- Python 버전 확인: Python 3.8+ 필요
