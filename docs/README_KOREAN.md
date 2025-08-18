# 문서 RAG 챗봇

복잡한 문서를 쉽게 이해할 수 있도록 도와주는 AI 챗봇입니다.

## 주요 기능

- **다양한 LLM 지원**: OpenAI, Google Gemini, Anthropic Claude, 로컬 LLM
- **문서 형식 지원**: TXT, Markdown, PDF (OCR 포함)
- **벡터 데이터베이스**: FAISS, ChromaDB
- **쉬운 설명 모드**: 전문 용어를 일반인도 이해할 수 있게 설명
- **진행률 표시**: 대용량 파일 처리 시 실시간 진행 상황 확인
- **디버그 모드**: 검색 결과 및 점수 확인 가능

## 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone <your-repo-url>
cd ragTest

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# OCR 지원 설치 (선택사항, 스캔된 PDF용)
bash scripts/setup/install_ocr.sh  # Windows는 수동 설치 필요
```

### 2. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 설정
# 최소 하나의 LLM/임베딩 API 키가 필요합니다:
# - UPSTAGE_API_KEY (임베딩, 권장)
# - OPENAI_API_KEY
# - GOOGLE_API_KEY  
# - ANTHROPIC_API_KEY
```

### 3. 실행

```bash
# 통합 실행 도구(권장)
python run_rag.py

# 또는 직접 실행
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속

## 사용 방법

### 1. 문서 업로드
- 좌측 사이드바에서 문서 파일 선택 (TXT, MD, PDF)
- "문서 처리 및 저장" 버튼 클릭
- 진행률 바를 통해 처리 상황 확인

### 2. LLM 모델 선택
- 사이드바에서 LLM 제공자 선택
- 원하는 모델 선택
- "모델 적용" 버튼 클릭

### 3. 질문하기
- 채팅창에 질문 입력
- AI가 업로드된 문서를 기반으로 답변
- 답변은 쉬운 언어로 설명됨

### 4. 고급 기능
- **검색 문서 수**: 더 많은 관련 문서를 검색하려면 슬라이더 조정
- **Temperature**: 답변의 창의성 조절 (낮을수록 일관성, 높을수록 창의성)
- **디버그 모드**: 검색 결과와 관련도 점수 확인

## 로컬 LLM 사용

로컬 LLM을 사용하려면 OpenAI 호환 API 서버가 필요합니다:

1. LM Studio, Ollama, 또는 유사한 도구로 로컬 모델 실행
2. `.env` 파일에서 설정:
   ```
   LOCAL_LLM_BASE_URL=http://localhost:1234
   LOCAL_LLM_MODEL=local-model
   ```
3. 앱에서 "로컬 LLM" 선택

## 문제 해결

### PDF 텍스트 추출 안됨
- OCR 옵션 활성화 (사이드바)
- Tesseract 설치 확인: `tesseract --version`

### 검색 결과 없음
- 문서가 제대로 업로드되었는지 확인 (문서 청크 수 확인)
- 다른 키워드로 검색
- 디버그 모드 활성화하여 상세 정보 확인

### API 키 오류
- `.env` 파일의 API 키 확인
- 해당 LLM 제공자의 API 키가 유효한지 확인

## 기여하기

버그 제보나 기능 제안은 이 저장소의 Issues에 등록해주세요.

## 라이선스

MIT License

## 문서 언어 정책

본 프로젝트는 한국어 문서를 기본으로 유지하며, 필요한 경우 영어판을 병행 관리합니다.
