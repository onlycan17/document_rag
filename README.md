# 🤖 RAG 기반 공공기관 정보시스템 챗봇

행정기관 및 공공기관 정보시스템 구축·운영 지침 문서를 활용한 RAG(Retrieval-Augmented Generation) 챗봇 시스템입니다.

## ✨ 주요 기능

- 📄 **다양한 문서 형식 지원**: TXT, Markdown, PDF (OCR 포함)
- 🔍 **고성능 벡터 검색**: FAISS 기반 효율적 문서 검색
- 💬 **자연스러운 한국어 대화**: 업스테이지 Solar 모델 최적화
- 🚀 **실시간 스트리밍 답변**: ChatGPT처럼 답변이 실시간으로 타이핑됨 ⚡
- 🚀 **다중 LLM 지원**: OpenRouter(기본), OpenAI, Google Gemini, Anthropic Claude (외부 API 전용)
- 📊 **체계적인 프로젝트 구조**: 모듈화된 코드 구조
- 🔧 **통합 관리 도구**: 메뉴 방식의 쉬운 실행 환경

## 🏗️ 프로젝트 구조

```
ragTest/
├── 📄 app.py                    # Streamlit 메인 애플리케이션
├── 📄 config.py                 # 시스템 설정
├── 📄 run_rag.py               # 🎯 통합 실행 스크립트 (메인)
├── 📄 requirements.txt          # Python 의존성
│
├── 📂 src/                     # 📦 소스 코드
│   ├── embeddings/             # 임베딩 모델
│   ├── loaders/                # 문서 로더
│   ├── rag/                    # RAG 체인 로직
│   ├── vectorstore/            # 벡터 데이터베이스
│   ├── models/                 # LLM 모델 관리
│   └── utils/                  # 유틸리티 함수
│
├── 📂 scripts/                 # 🔧 실행 스크립트
│   ├── setup/                  # 설치/설정 스크립트
│   │   ├── install_and_run.sh
│   │   ├── install_ocr.sh
│   │   └── setup.sh
│   ├── data_management/        # 데이터 관리
│   │   └── vector_db_manager.py  # 🎯 통합 벡터 DB 관리
│   └── run.sh                  # 앱 실행 스크립트
│
├── 📂 tests/                   # 🧪 테스트 코드
│   ├── debug/                  # 🔍 디버그 및 진단 테스트
│   │   ├── test_rag_context.py     # RAG 컨텍스트 검증
│   │   └── test_rag_query.py       # RAG 쿼리 진단
│   ├── legacy/                 # 📦 레거시 테스트
│   │   └── test_gpt41_rag.py       # GPT-4.1 특정 테스트
│   ├── test_simple.py          # 기본 기능 테스트
│   ├── test_upstage_embedding.py  # 임베딩 테스트
│   ├── test_keyword_expansion.py  # 키워드 확장 테스트
│   └── test_faiss_upstage_compatibility.py  # 호환성 테스트
│
├── 📂 docs/                    # 📚 문서
│   ├── INSTALL.md              # 설치 가이드
│   ├── UPSTAGE_SETUP_GUIDE.md  # 업스테이지 설정
│   └── README_KOREAN.md        # 한국어 상세 가이드
│
├── 📂 data/                    # 💾 데이터
│   ├── documents/              # 원본 문서
│   └── processed/              # 처리된 문서
│
└── 📂 logs/                    # 📋 로그 파일
```

## 🚀 빠른 시작

### 1️⃣ 저장소 클론
```bash
git clone [repository-url]
cd ragTest
```

### 2️⃣ 통합 실행 도구 사용 (권장)
```bash
python run_rag.py
```

**메뉴에서 선택할 수 있는 기능들:**
- 🚀 Streamlit 앱 실행
- 🗄️ 벡터 데이터베이스 관리
- 🧪 테스트 실행  
- ⚙️ 환경 설정
- ℹ️ 프로젝트 정보

### 3️⃣ 수동 설치 (고급 사용자)
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 API 키 설정

# 앱 실행
streamlit run app.py
```

## ⚙️ 환경 설정

### 필수: API 키 설정
`.env` 파일에 다음 중 **최소 하나의 API 키**를 설정하세요:

```env
# 업스테이지 API 키 (한국어 최적화, 권장)
UPSTAGE_API_KEY=your_upstage_api_key_here

# 기타 LLM API 키들 (선택)
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 선택: OCR 기능 (스캔된 PDF용)
```bash
# 자동 설치
bash scripts/setup/install_ocr.sh

# 또는 수동 설치 (macOS)
brew install tesseract tesseract-lang-kor poppler
```

## 📚 사용 방법

### 방법 1: 통합 실행 도구 (권장)
```bash
python run_rag.py
```
메뉴에서 원하는 기능을 선택하세요!

### 방법 2: 직접 실행
```bash
# 1. 벡터 DB 구축
python scripts/data_management/vector_db_manager.py

# 2. 앱 실행  
streamlit run app.py

# 3. 테스트 실행
python tests/test_simple.py
```

### 🚀 스트리밍 답변 사용법

**스트리밍 모드 (기본 활성화):**
- ✅ 답변이 **실시간으로 타이핑**되듯 나타남
- ✅ **검색 중**, **답변 생성 중** 등 상태 표시
- ✅ **ChatGPT와 동일한 사용자 경험**

**사이드바에서 토글 가능:**
- 🚀 **실시간 답변 (스트리밍)** 체크박스
- 체크 해제 시 기존 방식으로 동작 (한번에 전체 답변 표시)

**환경 변수로 기본값 설정:**
```bash
# .env 파일에 추가
ENABLE_STREAMING=true   # 스트리밍 활성화 (기본값)
ENABLE_STREAMING=false  # 스트리밍 비활성화
```

## 🔧 벡터 데이터베이스 관리

새로운 **통합 벡터 DB 관리 도구**를 사용하세요:

```bash
# 기본 사용법
python scripts/data_management/vector_db_manager.py

# 안전 모드 (OCR 비활성화)
python scripts/data_management/vector_db_manager.py --safe-mode

# 마크다운만 처리
python scripts/data_management/vector_db_manager.py --markdown-only

# 특정 파일들만 처리
python scripts/data_management/vector_db_manager.py --files "file1.pdf" "file2.md"

# 도움말 보기
python scripts/data_management/vector_db_manager.py --help
```

**주요 특징:**
- ✅ **3개 스크립트 통합**: 기존 중복 스크립트들을 하나로 통합
- ✅ **다양한 모드**: 전체/안전/마크다운 전용 모드 지원
- ✅ **상세한 진행률**: 실시간 처리 상황 표시
- ✅ **오류 복구**: 강화된 오류 처리 및 재시도 로직

## 🌞 업스테이지 임베딩(선택) 설치 안내

이 프로젝트는 업스테이지 임베딩(`langchain-upstage`)을 지원하지만, `requirements.txt`에는 포함하지 않습니다.  
이유: `langchain-upstage`가 `tokenizers<0.21`을 강제하여 최신 `transformers`와 충돌하기 때문입니다.

업스테이지를 사용할 경우 아래와 같이 별도 설치 후 `.env`에 `UPSTAGE_API_KEY`를 설정하세요.

```bash
pip install --no-deps langchain-upstage
```

## 🧪 테스트

```bash
# 통합 실행 도구에서 테스트 메뉴 선택
python run_rag.py

# 또는 직접 실행
python tests/test_simple.py          # 기본 기능 테스트
python tests/test_upstage_embedding.py  # 임베딩 테스트
```

## 📖 상세 문서

- 📋 **설치 가이드**: [docs/INSTALL.md](docs/INSTALL.md)
- ⚡ **업스테이지 설정**: [docs/UPSTAGE_SETUP_GUIDE.md](docs/UPSTAGE_SETUP_GUIDE.md)
- 🇰🇷 **한국어 가이드**: [docs/README_KOREAN.md](docs/README_KOREAN.md)

## 🛠️ 주요 개선사항 (v2.1)

### 🚀 실시간 스트리밍 답변 (NEW!)
- ✅ **ChatGPT 스타일 UI**: 답변이 실시간으로 타이핑되듯 나타남
- ✅ **상태 표시**: 검색 중, 생성 중 등 단계별 진행 상황 표시
- ✅ **토글 옵션**: 사이드바에서 스트리밍 온/오프 선택 가능
- ✅ **모든 LLM 지원**: OpenRouter, OpenAI, Google, Anthropic 외부 API 지원
- ✅ **에러 처리**: 스트리밍 중 오류 발생 시 안전한 복구

### 🏗️ 프로젝트 구조 재편
- ✅ **체계적인 디렉토리 구조**: 기능별 명확한 분리
- ✅ **스크립트 정리**: 설치/데이터관리/테스트 스크립트 분류
- ✅ **문서 통합**: 모든 가이드를 docs/ 디렉토리로 정리

### 🔧 통합 관리 도구
- ✅ **메인 실행 스크립트**: `run_rag.py` 메뉴 방식 통합 도구
- ✅ **벡터 DB 관리자**: 3개 중복 스크립트를 1개로 통합
- ✅ **설정 자동화**: 환경 설정 단계별 가이드

### 🧪 테스트 강화
- ✅ **테스트 패키지화**: tests/ 디렉토리로 체계적 관리
- ✅ **Import 경로 수정**: 새 구조에 맞는 경로 조정
- ✅ **실행 편의성**: 통합 도구에서 메뉴 선택으로 테스트 실행

## 🤝 기여하기

1. 이 저장소를 포크하세요
2. 기능 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🆘 문제 해결

### 자주 발생하는 문제들

1. **임베딩 모델 오류**
   ```bash
   # API 키 확인
   python tests/test_upstage_embedding.py
   ```

2. **OCR 관련 오류** 
   ```bash
   # 안전 모드로 실행
   python scripts/data_management/vector_db_manager.py --safe-mode
   ```

3. **Import 오류**
   ```bash
   # 가상환경 활성화 확인
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### 지원

- 🐛 **버그 리포트**: GitHub Issues 사용
- 💡 **기능 제안**: Discussion 탭 활용
- 📧 **직접 문의**: [이메일 주소]

---

**🎯 처음 사용자라면 `python run_rag.py`를 실행하고 4번 환경 설정부터 시작하세요!**
