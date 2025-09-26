# RAG 애플리케이션 성능 최적화 가이드

본 문서는 RAG 애플리케이션의 성능을 최적화하기 위한 실용적인 가이드를 제공합니다.

## 1. 성능 벤치마크 기준

### 현재 성능 지표 (기준선)
- **문서 처리 속도**: 평균 2-5초/문서 (PDF 기준)
- **검색 응답 시간**: 평균 200-500ms
- **메모리 사용량**: 문서 100개 기준 1-2GB
- **동시 사용자**: 1-2명 (단일 인스턴스)

### 목표 성능 지표
- **문서 처리 속도**: 1초 이내/문서 (60% 개선)
- **검색 응답 시간**: 100ms 이내 (70% 개선) 
- **메모리 사용량**: 문서 100개 기준 500MB 이하 (50% 감소)
- **동시 사용자**: 10명 이상 지원

## 2. 문서 처리 성능 최적화

### 2.1 PDF 처리 최적화

#### 현재 문제점
- [`_load_pdf_file`](src/loaders/document_loader_refactored.py:538) 메서드의 동기적 처리
- 이미지 추출과 텍스트 변환의 순차적 실행

#### 최적화 방안
```python
# 비동기 PDF 처리 구현
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncPDFProcessor:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_pdf_async(self, file_path: str) -> List[Document]:
        loop = asyncio.get_event_loop()
        
        # 이미지 추출과 텍스트 변환 병렬 실행
        image_task = loop.run_in_executor(self.executor, self._extract_images, file_path)
        text_task = loop.run_in_executor(self.executor, self._extract_text, file_path)
        
        images, text = await asyncio.gather(image_task, text_task)
        return self._combine_results(images, text)
```

#### 기대 효과
- 처리 속도 40% 이상 개선
- CPU 활용도 향상

### 2.2 청킹 알고리즘 최적화

#### 현재 문제점
- [`_enhanced_default_chunking`](src/loaders/document_loader_refactored.py:1377)의 재귀적 분할 오버헤드
- 의미 기반 청킹의 계산 복잡도

#### 최적화 방안
```python
class OptimizedChunkingStrategy:
    def __init__(self):
        self.sentence_detector = KoreanSentenceDetector()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        # 문장 단위 분할로 재귀적 분할 회피
        sentences = self._split_into_sentences(documents)
        
        # 동적 청크 크기 조정
        optimized_chunks = self._merge_sentences_optimally(sentences)
        return optimized_chunks
    
    def _split_into_sentences(self, documents: List[Document]) -> List[str]:
        # 한국어 문장 분할 최적화
        sentences = []
        for doc in documents:
            doc_sentences = self.sentence_detector.split(doc.page_content)
            sentences.extend(doc_sentences)
        return sentences
```

#### 기대 효과
- 청킹 속도 50% 이상 개선
- 청크 품질 향상

## 3. 벡터 검색 성능 최적화

### 3.1 하이브리드 검색 최적화

#### 현재 문제점
- [`_hybrid_search`](src/vectorstore/vector_db.py:206)에서 벡터/키워드 검색 순차 실행
- [`_combine_search_results`](src/vectorstore/vector_db.py:281)의 O(n²) 연산

#### 최적화 방안
```python
class OptimizedHybridSearch:
    async def search_async(self, query: str, k: int) -> List[Tuple[Document, float]]:
        # 벡터 검색과 키워드 검색 병렬 실행
        vector_task = asyncio.create_task(self._vector_search_async(query, k))
        keyword_task = asyncio.create_task(self._keyword_search_async(query, k))
        
        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)
        
        # 효율적인 결과 통합 (O(n log n))
        return self._optimized_combine_results(vector_results, keyword_results)
    
    def _optimized_combine_results(self, vector_results, keyword_results):
        # 해시맵을 이용한 O(1) 조회로 성능 개선
        results_map = {}
        
        for doc, score in vector_results:
            doc_id = self._get_doc_id(doc)
            results_map[doc_id] = {'doc': doc, 'vector_score': score}
        
        for doc, score in keyword_results:
            doc_id = self._get_doc_id(doc)
            if doc_id in results_map:
                results_map[doc_id]['keyword_score'] = score
            else:
                results_map[doc_id] = {'doc': doc, 'keyword_score': score}
        
        return self._calculate_combined_scores(results_map)
```

#### 기대 효과
- 검색 응답 시간 60% 이상 개선
- CPU 사용량 감소

### 3.2 캐싱 전략 구현

#### Redis 캐싱 계층
```python
import redis
import json
import hashlib

class SearchCache:
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl
    
    def _get_cache_key(self, query: str, k: int) -> str:
        # 쿼리 기반 캐시 키 생성
        key_data = f"{query}:{k}"
        return f"search:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def get_cached_results(self, query: str, k: int) -> Optional[List[Document]]:
        cache_key = self._get_cache_key(query, k)
        cached = self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        return None
    
    async def set_cached_results(self, query: str, k: int, results: List[Document]):
        cache_key = self._get_cache_key(query, k)
        self.redis.setex(cache_key, self.default_ttl, json.dumps(results))
```

#### 기대 효과
- 반복 쿼리 응답 시간 80% 이상 개선
- 백엔드 부하 감소

## 4. 메모리 사용량 최적화

### 4.1 문서 캐시 최적화

#### 현재 문제점
- [`documents_cache`](src/vectorstore/vector_db.py:38)가 모든 문서를 메모리에 보관
- TF-IDF 매트릭스의 메모리 점유율 높음

#### 최적화 방안
```python
class MemoryOptimizedVectorDB:
    def __init__(self):
        self.documents_cache = []  # 메타데이터만 저장
        self.content_store = DiskBackedContentStore()  # 내용은 디스크 저장
    
    def add_documents(self, documents: List[Document]):
        for doc in documents:
            # 내용은 디스크에 저장, 메타데이터만 메모리 보관
            content_id = self.content_store.store(doc.page_content)
            lightweight_doc = Document(
                page_content=content_id,  # 참조만 저장
                metadata=doc.metadata
            )
            self.documents_cache.append(lightweight_doc)
```

#### 기대 효과
- 메모리 사용량 60% 이상 감소
- 대용량 문서 처리 가능

### 4.2 임베딩 배치 처리

#### 현재 문제점
- 문서별 개별 임베딩 생성의 오버헤드

#### 최적화 방안
```python
class BatchEmbeddingProcessor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    async def embed_documents_batch(self, documents: List[Document]) -> List[List[float]]:
        # 문서를 배치로 그룹화
        batches = [documents[i:i + self.batch_size] 
                  for i in range(0, len(documents), self.batch_size)]
        
        embeddings = []
        for batch in batches:
            batch_texts = [doc.page_content for doc in batch]
            batch_embeddings = await self._embed_batch_async(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return embeddings
```

#### 기대 효과
- 임베딩 생성 속도 70% 이상 개선
- API 호출 횟수 감소

## 5. 데이터베이스 최적화

### 5.1 FAISS 인덱스 최적화

#### 인덱스 파라미터 튜닝
```python
class OptimizedFAISSIndex:
    def __init__(self):
        self.index = faiss.IndexHNSWFlat(768, 32)  # HNSW 알고리즘 사용
        self.index.hnsw.efConstruction = 200  # 빌드 품질 향상
        self.index.hnsw.efSearch = 100  # 검색 품질 향상
    
    def optimize_for_search(self):
        # 검색 최적화를 위한 인덱스 튜닝
        faiss.omp_set_num_threads(4)  # 병렬 처리
        self.index.make_direct_map()  # 직접 맵 생성으로 검색 속도 향상
```

### 5.2 ChromaDB 최적화

#### 설정 최적화
```python
# ChromaDB 클라이언트 설정
client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

# 컬렉션 생성 시 최적화 설정
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:ef_construction": 200}
)
```

## 6. 모니터링 및 프로파일링

### 6.1 성능 메트릭스 수집

```python
import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceMetrics:
    processing_time: float
    memory_usage: float
    search_latency: float
    cache_hit_rate: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
    
    def track_operation(self, operation: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                self.metrics[operation] = PerformanceMetrics(
                    processing_time=end_time - start_time,
                    memory_usage=end_memory - start_memory,
                    search_latency=0,  # 검색 연산별 측정
                    cache_hit_rate=0   # 캐시별 측정
                )
                
                return result
            return wrapper
        return decorator
```

### 6.2 프로파일링 도구 활용

```bash
# CPU 프로파일링
python -m cProfile -o profile_stats.prof app.py

# 메모리 프로파일링
python -m memory_profiler app.py

# 시각화 도구
snakeviz profile_stats.prof
```

## 7. 배포 환경 최적화

### 7.1 Docker 최적화
```dockerfile
# 다단계 빌드로 이미지 크기 최소화
FROM python:3.11-slim as builder

# 프로덕션 이미지
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 메모리 제한 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 성능 최적화 파라미터
CMD ["python", "-O", "app.py"]  # 바이트코드 최적화
```

### 7.2 웹 서버 최적화
```python
# Gunicorn 설정 (gunicorn.conf.py)
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
max_requests = 1000
max_requests_jitter = 100
timeout = 120
```

## 8. 성능 테스트 체크리스트

### 정기적 성능 테스트
- [ ] 문서 처리 속도 측정 (100개 문서 기준)
- [ ] 검색 응답 시간 측정 (다양한 쿼리)
- [ ] 메모리 사용량 모니터링
- [ ] 동시 사용자 부하 테스트
- [ ] 캐시 효율성 분석

### 성능 회고 주기
- 주간: 주요 지표 점검
- 월간: 심층 성능 분석
- 분기별: 아키텍처 재평가

이 가이드를 따라 체계적으로 성능 최적화를 진행하면 RAG 애플리케이션의 전반적인 성능을 크게 향상시킬 수 있습니다.