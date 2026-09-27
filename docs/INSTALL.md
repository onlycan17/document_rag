# 🚀 설치 가이드

이 문서는 RAG 챗봇 시스템을 로컬 환경에 설치하고 실행하는 방법을 안내합니다.

## 📋 사전 요구사항

- **Python 3.10 이상**: 시스템에 Python이 설치되어 있어야 합니다.
- **Git**: 소스 코드를 클론하기 위해 필요합니다.

## ⚙️ 설치 절차

### 1. 저장소 클론

먼저, 터미널을 열고 Git을 사용하여 프로젝트 저장소를 클론합니다.

```bash
git clone [repository-url]
cd ragTest
```

### 2. 가상환경 생성 및 활성화

프로젝트 의존성을 시스템의 다른 Python 환경과 격리하기 위해 가상환경을 사용하는 것을 강력히 권장합니다.

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 의존성 설치

`requirements.txt` 파일에 명시된 모든 Python 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

API 키와 같은 민감한 정보를 관리하기 위해 `.env` 파일을 설정해야 합니다. 제공된 예제 파일을 복사하여 시작하세요.

```bash
cp .env.example .env
```

이제 텍스트 편집기로 `.env` 파일을 열고, 사용할 LLM 서비스의 API 키를 입력하세요. **최소 하나 이상의 API 키가 필요합니다.**

```env
# --- 필수 (최소 1개 이상) ---
# 업스테이지 API 키 (한국어 모델에 권장)
UPSTAGE_API_KEY="your_upstage_api_key"

# OpenAI API 키
OPENAI_API_KEY="your_openai_api_key"

# Google Gemini API 키
GOOGLE_API_KEY="your_google_api_key"

# Anthropic Claude API 키
ANTHROPIC_API_KEY="your_anthropic_api_key"

# --- 선택 (고급 기능용) ---
# OpenRouter API 키 (지능형 이미지 분석용)
OPNEROUTER_API_KEY="your_openrouter_api_key"
```

### 5. (선택) OCR 기능 설치

스캔된 PDF(텍스트 레이어가 없는 이미지 PDF)는 자동으로 OCR로 처리됩니다.

엔진 우선순위는 `.env`의 `OCR_ENGINE`(기본 `upstage`)이며, 실패하거나 사용할 수 없으면 Vision → Tesseract 순으로 자동 전환됩니다.

- **Upstage Document Parse (기본)**: `UPSTAGE_API_KEY`만 있으면 됩니다. 표 구조를 보존하고 머리말·쪽번호를 제외합니다. 비용 $0.01/쪽(220쪽 약 $2.2).
- **macOS Vision (무료 폴백)**: `requirements.txt`의 `pyobjc-framework-Vision`으로 추가 설치 없이 동작합니다. 기울임꼴 한글도 인식하지만 표는 줄 단위로 풀립니다.
- **Tesseract (그 외 OS 폴백)**: Tesseract OCR 엔진과 한국어 언어팩이 필요합니다.

```bash
# 자동 설치 스크립트 실행 (macOS/Linux)
bash scripts/setup/install_ocr.sh

# 또는 수동 설치 (macOS 예시)
brew install tesseract tesseract-lang
```

## ✅ 설치 확인 및 실행

모든 설치가 완료되면, 통합 실행 도구를 사용하여 시스템을 시작할 수 있습니다.

```bash
python run_rag.py
```

메뉴에서 "Streamlit 앱 실행"을 선택하여 웹 UI를 시작하세요. 잠시 후 웹 브라우저에서 챗봇 애플리케이션이 열립니다.

---

**🎉 이제 모든 준비가 완료되었습니다!** 챗봇을 사용해 보세요.