# RAG 애플리케이션 리팩토링 가이드

본 문서는 RAG 애플리케이션의 아키텍처 개선을 위한 단계별 리팩토링 가이드를 제공합니다.

## 1. 리팩토링 목표

### 주요 목표
- **모듈화**: 단일 책임 원칙(SRP) 적용으로 컴포넌트 분리
- **성능 개선**: 비동기 처리 및 캐싱 도입
- **확장성**: 마이크로서비스 아키텍처로 전환
- **유지보수성**: 테스트 용이성 및 디버깅 개선

### 성과 지표
- 문서 처리 속도 60% 이상 개선
- 검색 응답 시간 70% 이상 개선
- 코드 복잡도 40% 이상 감소
- 테스트 커버리지 80% 이상 달성

## 2. 1단계: 즉시 개선 (High Impact, Low Effort)

### 2.1 문서 처리 모듈 분리

#### 현재 문제점
- [`EnhancedDocumentLoader`](src/loaders/document_loader_refactored.py:26) 클래스가 1500라인 이상으로 과도하게 복잡
- 단일 메서드가 300라인 이상으로 단일 책임 원칙 위반

#### 개선 방안
```python
# 새로운 클래스 구조
class DocumentProcessor:
    """문서 처리 파이프라인 오케스트레이터"""
    pass

class PDFProcessor:
    """PDF 전용 처리기 (OCR, 이미지 추출 등)"""
    pass

class MarkdownProcessor:
    """마크다운 전용 처리기"""
    pass

class TextProcessor:
    """일반 텍스트 처리기"""
    pass

class ChunkingStrategy:
    """청킹 전략 인터페이스"""
    pass
```

#### 구현 단계
1. [`_load_pdf_file`](src/loaders/document_loader_refactored.py:538) 메서드를 PDFProcessor 클래스로 분리
2. 마크다운 처리 로직을 MarkdownProcessor 클래스로 분리
3. 청킹 전략을 전략 패턴으로 구현
4. 팩토리 패턴을 통한 문서 처리기 생성

### 2.2 비동기 처리 도입

#### 현재 문제점
- 문서 업로드 시 동기적 처리로 UI 블로킹 발생
- 대용량 PDF 처리 시 사용자 경험 저하

#### 개선 방안
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncDocumentProcessor:
    async def process_document_async(self, file_path: str) -> List[Document]:
        # 비동기 문서 처리
        pass
    
    async def batch_process_async(self, file_paths: List[str]) -> List[Document]:
        # 배치 비동기 처리
        pass
```

#### 구현 단계
1. Streamlit의 `st.rerun()` 대신 `st.empty()`와 상태 표시기 사용
2. 백그라운드 스레드 풀을 이용한 문서 처리
3. 진행률 표시를 위한 이벤트 시스템 도입

### 2.3 캐시 계층 추가

#### 현재 문제점
- 반복적인 검색 쿼리에 대한 캐싱 부재
- 문서 처리 결과 캐싱 미지원

#### 개선 방안
```python
import redis
from functools import lru_cache

class SearchCache:
    """검색 결과 캐싱 시스템"""
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
    
    def get_cached_results(self, query: str, k: int) -> Optional[List[Document]]:
        # Redis에서 캐시된 결과 조회
        pass
    
    def set_cached_results(self, query: str, k: int, results: List[Document], ttl: int = 3600):
        # Redis에 결과 캐싱
        pass
```

#### 구현 단계
1. Redis 클라이언트 설정 및 연결
2. 검색 결과 캐싱 메커니즘 구현
3. TTL(Time To Live) 설정으로 캐시 관리
4. 캐시 무효화 전략 수립

## 3. 2단계: 중기 개선 (Medium Impact, Medium Effort)

### 3.1 마이크로서비스 아키텍처 설계

#### 서비스 분리 계획
1. **Document Processing Service**
   - 문서 업로드, 전처리, 청킹 담당
   - REST API 또는 gRPC 인터페이스 제공

2. **Vector DB Service** 
   - 벡터 인덱싱 및 검색 담당
   - 하이브리드 검색 엔진 포함

3. **Search Service**
   - 검색 쿼리 처리 및 결과 통합
   - 캐싱 및 성능 최적화

4. **API Gateway**
   - 통합 엔드포인트 제공
   - 인증, 로드 밸런싱, 모니터링

### 3.2 설정 관리 중앙화

#### 현재 문제점
- 설정이 여러 파일에 분산되어 있음
- 환경별 설정 관리 어려움

#### 개선 방안
```python
# config/central_config.py
class CentralConfig:
    """중앙 설정 관리 시스템"""
    
    def __init__(self):
        self.config_map = {
            'development': DevelopmentConfig,
            'production': ProductionConfig,
            'testing': TestingConfig
        }
    
    def get_config(self, environment: str) -> BaseConfig:
        return self.config_map[environment]()
```

## 4. 3단계: 장기 개선 (High Impact, High Effort)

### 4.1 분산 벡터 데이터베이스

#### 구현 계획
- 여러 벡터 DB 샤드에 문서 분산 저장
- 샤딩 키 기반 문서 배치
- 쿼리 라우팅 및 결과 병합

### 4.2 머신러닝 파이프라인

#### 기능 계획
- 문서 품질 자동 평가
- 최적 청킹 전략 추천
- 검색 결과 품질 개선

## 5. 리팩토링 우선순위 매트릭스

| 우선순위 | 작업 | 예상 소요시간 | 기대 효과 |
|---------|------|---------------|-----------|
| 높음 | 문서 처리 모듈 분리 | 2-3일 | 코드 가독성 60% 개선 |
| 높음 | 비동기 처리 도입 | 3-4일 | UI 응답성 70% 개선 |
| 중간 | 캐시 계층 추가 | 2일 | 검색 성능 50% 개선 |
| 중간 | 마이크로서비스 설계 | 1주 | 확장성 향상 |
| 낮음 | 분산 벡터 DB | 2주 | 대용량 처리 능력 향상 |

## 6. 테스트 전략

### 단위 테스트
- 각 모듈별 독립적 테스트
- Mock 객체를 이용한 의존성 분리

### 통합 테스트
- 서비스 간 통신 테스트
- end-to-end 시나리오 검증

### 성능 테스트
- 부하 테스트 및 성능 벤치마킹
- 메모리 사용량 및 응답 시간 모니터링

## 7. 마이그레이션 계획

### 점진적 마이그레이션
1. 기존 코드와 새 코드 병행 운영
2. 기능별 점진적 전환
3. 롤백 계획 수립

### 데이터 마이그레이션
- 벡터 인덱스 변환 전략
- 문서 메타데이터 보존
- 호환성 유지 보장

## 8. 모니터링 및 유지보수

### 모니터링 지표
- 처리량(Throughput) 및 지연 시간(Latency)
- 에러율 및 성공률
- 리소스 사용량(CPU, Memory, Disk)

### 로깅 전략
- 구조화된 로깅(JSON 형식)
- 로그 수집 및 분석 시스템 연동
- 실시간 알림 설정

이 가이드를 따라 단계별로 리팩토링을 진행하면 RAG 애플리케이션의 성능, 확장성, 유지보수성을 크게 향상시킬 수 있습니다.