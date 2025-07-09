# RAG 기반 공공기관 정보시스템 챗봇

행정기관 및 공공기관 정보시스템 구축·운영 지침 문서를 활용한 RAG(Retrieval-Augmented Generation) 챗봇 시스템입니다.

## 주요 기능

- 📄 다양한 문서 형식 지원 (TXT, MD, PDF)
- 🔍 벡터 데이터베이스를 활용한 효율적인 검색
- 💬 자연스러운 한국어 대화형 인터페이스
- 🚀 OpenAI API 및 로컬 LLM 지원
- 📊 확장 가능한 모듈식 구조

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd ragTest
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 열어 필요한 API 키 설정
```

## 사용 방법

1. Streamlit 앱 실행
```bash
streamlit run app.py
```

2. 브라우저에서 http://localhost:8501 접속

3. 사이드바에서 문서 업로드 또는 domain.md 파일 로드

4. 채팅 인터페이스에서 질문 입력

## 프로젝트 구조

```
ragTest/
├── app.py                 # Streamlit 메인 애플리케이션
├── config.py             # 설정 관리
├── requirements.txt      # 패키지 의존성
├── .env.example         # 환경 변수 예시
├── src/
│   ├── loaders/         # 문서 로더 모듈
│   ├── embeddings/      # 임베딩 모델 모듈
│   ├── vectorstore/     # 벡터 데이터베이스 모듈
│   ├── rag/            # RAG 체인 모듈
│   └── utils/          # 유틸리티 함수
├── data/
│   ├── documents/      # 원본 문서 저장
│   └── processed/      # 처리된 문서 저장
└── vector_db/          # 벡터 데이터베이스 저장

```

## 설정 옵션

### LLM 선택
- OpenAI GPT 모델 (API 키 필요)
- 로컬 LLM (Ollama 등)

### 벡터 데이터베이스
- ChromaDB (기본값)
- FAISS

### 임베딩 모델
- OpenAI text-embedding-ada-002
- Sentence Transformers (다국어 지원)

## 확장성

- 새로운 문서 형식 추가 가능
- 다양한 LLM 모델 통합 가능
- 커스텀 프롬프트 템플릿 설정
- 배치 처리 및 대용량 문서 지원

## 주의사항

- OpenAI API 사용 시 API 키가 필요합니다
- 로컬 LLM 사용 시 Ollama 등의 서버가 실행 중이어야 합니다
- 대용량 문서 처리 시 충분한 메모리가 필요합니다