# 업스테이지 Solar Embedding 모델 설정 가이드

## 🚀 개요

업스테이지의 `solar-embedding-1-large-query` 모델을 성공적으로 통합했습니다!  
이 가이드는 모델 설정부터 벡터 데이터베이스 재구축까지의 전체 과정을 안내합니다.

## 📋 설정 단계

### 1단계: 업스테이지 API 키 발급

1. **업스테이지 콘솔 접속**
   - 🌐 [https://console.upstage.ai/](https://console.upstage.ai/)
   - 회원가입 또는 로그인

2. **API 키 생성**
   - API 키 생성 페이지로 이동
   - 새 API 키 생성
   - **생성된 키를 안전한 곳에 복사 보관**

### 2단계: 환경 변수 설정

1. **`.env` 파일 생성**
   ```bash
   # 프로젝트 루트에서 실행
   cp .env.example .env
   ```

2. **API 키 설정**
   ```bash
   # .env 파일을 열어서 다음과 같이 설정
   UPSTAGE_API_KEY=your_actual_api_key_here
   ```

   **⚠️ 주의사항:**
   - `your_actual_api_key_here` 부분을 실제 발급받은 API 키로 교체
   - API 키는 절대 공개하지 마세요
   - `.env` 파일은 git에 커밋하지 마세요

### 3단계: 패키지 설치 확인

필요한 패키지가 모두 설치되었는지 확인:

```bash
pip install langchain-upstage
```

## 🔧 벡터 데이터베이스 재구축

### 안전한 방법 (권장)

멀티프로세싱 오류를 방지하는 안전한 스크립트 사용:

```bash
python safe_rebuild_vector_db.py
```

**특징:**
- ✅ OCR 기능 비활성화로 안정성 확보
- ✅ 기본 PyPDFLoader만 사용
- ✅ 멀티프로세싱 오류 방지
- ✅ 상세한 진행 상황 표시

### 기존 방법 (고급 사용자용)

OCR 기능을 포함한 전체 기능 사용:

```bash
# 기존 벡터 DB 삭제
rm -rf vector_db/

# 재구축 실행
python rebuild_vector_db.py
```

## 🧪 테스트

### 임베딩 모델 테스트

```bash
python test_upstage_embedding.py
```

**예상 출력:**
```
✅ 모델 정보:
   - 타입: upstage
   - 모델명: solar-embedding-1-large-query
   - 차원: 4096

📊 단일 쿼리 임베딩 테스트:
   임베딩 차원: 4096
   임베딩 샘플: [0.1234, -0.5678, ...] (처음 5개 값)

✅ 모든 테스트가 성공적으로 완료되었습니다!
```

### RAG 시스템 테스트

```bash
python test_simple.py
```

## 📊 모델 정보

### 업스테이지 Solar Embedding 1 Large Query

- **모델명**: `solar-embedding-1-large-query`
- **차원**: 4096 (기존 1536에서 크게 증가)
- **언어**: 한국어/영어 모두 뛰어난 성능
- **최적화**: 쿼리 검색에 특화된 모델
- **성능**: 한국어 임베딩 업계 최고 수준

### 기대 효과

1. **🎯 향상된 검색 정확도**
   - 4096 차원으로 더 정밀한 의미 표현
   - 미세한 의미 차이까지 포착

2. **🇰🇷 한국어 특화 성능**
   - 한국어 문서에 대한 우수한 이해
   - 한국어 쿼리-문서 매칭 성능 향상

3. **⚡ 효율적인 클라우드 서비스**
   - 로컬 GPU 리소스 절약
   - 업스테이지 최적화된 인프라 활용

## 🔍 문제 해결

### 자주 발생하는 문제

1. **API 키 오류**
   ```
   ❌ UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다.
   ```
   **해결**: `.env` 파일에 올바른 API 키 설정 확인

2. **차원 불일치 오류**
   ```
   ❌ 벡터 차원이 맞지 않습니다.
   ```
   **해결**: 기존 벡터 DB 삭제 후 재구축

3. **PDF 로딩 실패**
   ```
   ❌ cannot pickle '_thread.RLock' object
   ```
   **해결**: `safe_rebuild_vector_db.py` 사용

4. **메모리 부족**
   ```
   ❌ OutOfMemoryError
   ```
   **해결**: 대용량 파일을 작은 단위로 분할

### 로그 확인

문제 발생 시 로그 파일 확인:
```bash
tail -f logs/app.log
```

## 📈 성능 최적화

### 설정 조정

`config.py`에서 다음 설정을 조정할 수 있습니다:

```python
# 청크 크기 조정 (기본: 1200)
chunk_size: int = 1200

# 오버랩 크기 조정 (기본: 200)  
chunk_overlap: int = 200

# 검색 결과 수 조정 (기본: 8)
k_documents: int = 8

# 검색 임계값 조정
search_threshold_faiss: float = 1.24
```

### 모니터링

벡터 DB 상태 확인:
```bash
# 벡터 DB 디렉토리 크기
du -sh vector_db/

# 저장된 문서 수 확인 (로그에서)
grep "청크" logs/app.log | tail -10
```

## 🎉 완료!

모든 설정이 완료되면 웹 인터페이스를 통해 향상된 RAG 시스템을 사용할 수 있습니다:

```bash
streamlit run app.py
```

**새로운 기능:**
- 🔍 더 정확한 한국어 검색
- 📚 향상된 문서 이해도
- ⚡ 빠른 응답 속도
- 🎯 정밀한 의미 매칭

---

## 📞 지원

추가 도움이 필요하면:
- 📚 [업스테이지 문서](https://developers.upstage.ai/)
- 🔧 `test_upstage_embedding.py` 실행하여 진단
- 💬 로그 파일 확인: `logs/` 디렉터리 