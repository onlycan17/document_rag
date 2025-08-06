from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_upstage import UpstageEmbeddings
from config import settings
from src.utils import TextProcessor
import numpy as np
import logging
import time
import random
from functools import wraps

logger = logging.getLogger(__name__)

def api_retry_with_backoff(max_retries=None, base_delay=None, max_delay=None):
    """
    API 호출 재시도 데코레이터
    - 429 오류 시 지수 백오프로 재시도
    - 네트워크 오류 시 재시도
    - Rate Limit 대응을 위해 더 관대한 재시도 정책 적용
    """
    # config에서 기본값 가져오기
    if max_retries is None:
        max_retries = settings.api_max_retries
    if base_delay is None:
        base_delay = settings.api_base_delay
    if max_delay is None:
        max_delay = settings.api_max_delay
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_message = str(e)
                    
                    # 429 오류 (Too Many Requests) 확인
                    if "429" in error_message or "too_many_requests" in error_message.lower() or "rate limit" in error_message.lower():
                        if attempt < max_retries:
                            # 지수 백오프 계산 (랜덤 지터 포함)
                            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                            logger.warning(f"⚠️ API 요청 제한 초과 (Rate Limit 429 오류)")
                            logger.warning(f"📊 재시도 정보: {attempt + 1}/{max_retries + 1}번째 시도")
                            logger.warning(f"⏰ {delay:.1f}초 후 재시도합니다...")
                            logger.info(f"💡 현재 설정: max_retries={max_retries}, base_delay={base_delay}, max_delay={max_delay}")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"🚨 API Rate Limit 재시도 횟수 초과: {max_retries + 1}번 모두 실패")
                            logger.error("💡 해결 방법:")
                            logger.error("   1. 잠시 후 다시 시도해보세요")
                            logger.error("   2. 환경변수로 재시도 설정 조정: API_MAX_RETRIES, API_BASE_DELAY, API_MAX_DELAY")
                            logger.error("   3. API 제공업체에서 Rate Limit 증가 요청 고려")
                    
                    # 네트워크 관련 오류 확인
                    elif any(keyword in error_message.lower() for keyword in 
                            ["connection", "timeout", "network", "temporary", "unavailable", "service"]):
                        if attempt < max_retries:
                            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), max_delay)
                            logger.warning(f"🌐 네트워크/서비스 오류 발생: {error_message[:100]}...")
                            logger.warning(f"📊 재시도 정보: {attempt + 1}/{max_retries + 1}번째 시도")
                            logger.warning(f"⏰ {delay:.1f}초 후 재시도합니다...")
                            time.sleep(delay)
                            continue
                    
                    # 재시도 불가능한 오류면 즉시 발생
                    raise e
            
            # 모든 재시도가 실패한 경우
            logger.error(f"🚨 API 호출이 {max_retries + 1}번 모두 실패했습니다")
            logger.error(f"❌ 최종 오류: {str(last_exception)[:200]}...")
            logger.error("💡 Rate Limit 문제인 경우 다음을 시도해보세요:")
            logger.error("   • 잠시 후 다시 실행")
            logger.error("   • .env 파일에 API_MAX_RETRIES=10, API_MAX_DELAY=600 설정")
            logger.error("   • API 제공업체 Rate Limit 증가 요청")
            raise last_exception
        
        return wrapper
    return decorator

class EmbeddingModel:
    """
    향상된 임베딩 모델 클래스
    - 한국어 특화 모델 지원
    - 모델 성능 최적화
    - 배치 처리 지원
    """
    
    def __init__(self):
        # 한국어 특화 모델 정보
        self.korean_models = {
            "kosentencebert": "jhgan/ko-sbert-nli",
            "kosentencebert_multitask": "jhgan/ko-sbert-multitask",
            "kobert": "klue/bert-base",
            "roberta_korean": "klue/roberta-base",
            "multilingual_optimized": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
        }
        
        if settings.embedding_provider == "openai" and settings.openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model="text-embedding-ada-002"
            )
            self.model_type = "openai"
            logger.info("OpenAI 임베딩 모델 초기화 완료")
        elif settings.embedding_provider == "upstage" and settings.upstage_api_key:
            # 업스테이지 solar-embedding-1-large-query 모델 초기화
            self.embeddings = UpstageEmbeddings(
                api_key=settings.upstage_api_key,
                model=settings.upstage_embedding_model
            )
            self.model_type = "upstage"
            logger.info(f"업스테이지 임베딩 모델 초기화 완료: {settings.upstage_embedding_model}")
        else:
            # 한국어 성능이 우수한 모델 우선 선택
            model_name = self._select_best_korean_model()
            
            # show_progress_bar 오류 방지를 위한 개선된 초기화
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        'device': 'cpu',
                        'trust_remote_code': True
                    },
                    encode_kwargs={
                        'normalize_embeddings': True,
                        'batch_size': 32,  # 배치 크기 최적화
                    }  # show_progress_bar 매개변수 제거
                )
            except Exception as e:
                logger.warning(f"HuggingFaceEmbeddings 초기화 실패, 기본 설정으로 재시도: {str(e)}")
                # 더 간단한 설정으로 재시도
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'}
                )
            
            self.model_type = "huggingface"
            logger.info(f"한국어 최적화 임베딩 모델 초기화 완료: {model_name}")
    
    def _select_best_korean_model(self) -> str:
        """
        사용 가능한 최적의 한국어 모델 선택
        설정에서 지정된 모델이 있으면 우선 사용하고,
        없으면 한국어 성능이 좋은 모델 순으로 시도
        """
        # 설정에서 모델이 지정된 경우
        if hasattr(settings, 'korean_embedding_model') and settings.korean_embedding_model:
            return settings.korean_embedding_model
        
        # 한국어 성능 우선순위에 따른 모델 선택
        priority_models = [
            "jhgan/ko-sbert-multitask",  # 다양한 태스크에 최적화
            "jhgan/ko-sbert-nli",        # 한국어 자연어 추론에 특화
            settings.embedding_model_name  # 기본 설정 모델
        ]
        
        for model_name in priority_models:
            try:
                # 모델 로드 테스트
                test_embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # 간단한 테스트
                test_embeddings.embed_query("테스트")
                logger.info(f"선택된 한국어 임베딩 모델: {model_name}")
                return model_name
            except Exception as e:
                logger.warning(f"모델 {model_name} 로드 실패, 다음 모델 시도: {str(e)}")
                continue
        
        # 모든 모델이 실패한 경우 기본 모델 사용
        logger.warning("한국어 특화 모델 로드 실패, 기본 모델 사용")
        return settings.embedding_model_name
    
    @api_retry_with_backoff()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서를 배치로 임베딩
        - 빈 텍스트 필터링
        - 배치 처리 최적화
        - API 요청 제한 오류 재시도 지원
        """
        # 빈 텍스트 제거 및 전처리
        filtered_texts = []
        text_mapping = []  # 원본 인덱스 매핑
        
        for i, text in enumerate(texts):
            cleaned_text = self._preprocess_text(text)
            if cleaned_text:
                filtered_texts.append(cleaned_text)
                text_mapping.append(i)
        
        if not filtered_texts:
            return [[0.0] * self.get_embedding_dimension()] * len(texts)
        
        # 임베딩 생성
        try:
            embeddings = self.embeddings.embed_documents(filtered_texts)
        except Exception as e:
            # 다양한 오류 상황 처리
            if "show_progress_bar" in str(e) and self.model_type == "huggingface":
                # HuggingFace 모델의 경우 직접 encode 메서드 호출
                try:
                    model = self.embeddings.client
                    embeddings = model.encode(filtered_texts, show_progress_bar=False)
                    embeddings = [emb.tolist() for emb in embeddings]
                except Exception:
                    # 더 안전한 방법으로 재시도
                    model = self.embeddings.client
                    embeddings = model.encode(filtered_texts)
                    embeddings = [emb.tolist() for emb in embeddings]
            else:
                raise e
        
        # 원본 순서에 맞게 결과 정렬
        result = []
        embedding_index = 0
        
        for i in range(len(texts)):
            if i in text_mapping:
                result.append(embeddings[embedding_index])
                embedding_index += 1
            else:
                # 빈 텍스트는 제로 벡터
                result.append([0.0] * len(embeddings[0]) if embeddings else [0.0] * self.get_embedding_dimension())
        
        return result
    
    @api_retry_with_backoff()
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리를 임베딩
        - 쿼리 전처리 및 확장
        - API 요청 제한 오류 재시도 지원
        """
        # 쿼리 전처리
        processed_query = self._preprocess_query(text)
        
        if not processed_query:
            return [0.0] * self.get_embedding_dimension()
        
        try:
            return self.embeddings.embed_query(processed_query)
        except Exception as e:
            # 다양한 오류 상황 처리
            if "show_progress_bar" in str(e) and self.model_type == "huggingface":
                # HuggingFace 모델의 경우 직접 encode 메서드 호출
                try:
                    model = self.embeddings.client
                    embeddings = model.encode([processed_query], show_progress_bar=False)
                    return embeddings[0].tolist()
                except Exception:
                    # 더 안전한 방법으로 재시도
                    model = self.embeddings.client
                    embeddings = model.encode([processed_query])
                    return embeddings[0].tolist()
            else:
                raise e
    
    def _preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리
        - 불필요한 공백 제거
        - 특수 문자 정리
        - 너무 짧은 텍스트 필터링
        """
        cleaned = TextProcessor.clean_text(text, remove_stopwords=False)
        return cleaned if TextProcessor.is_valid_text(cleaned, min_length=10) else ""
    
    def _preprocess_query(self, query: str) -> str:
        """
        쿼리 전처리 및 확장
        - 기본 전처리
        - 한국어 쿼리 최적화
        """
        if not query:
            return ""
        
        # 기본 전처리
        query = self._preprocess_text(query)
        
        # 기본적인 쿼리 확장만 사용 (임베딩에서는 과도한 확장 회피)
        # RAG의 _preprocess_query에서는 전체 확장을 사용
        words = query.split()
        expanded_words = []
        
        # 비교적 단순한 확장만 수행
        simple_expansions = {
            "몽촌토성": "몽촌토성 몽촌 토성",
            "백제": "백제 한성백제",
            "시스템": "시스템 정보시스템"
        }
        
        for word in words:
            if word in simple_expansions:
                expanded_words.extend(simple_expansions[word].split())
            else:
                expanded_words.append(word)
        
        # 중복 제거
        unique_words = []
        seen = set()
        for word in expanded_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        return ' '.join(unique_words[:10])  # 최대 10개 단어로 제한
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        if self.model_type == "openai":
            return 1536  # OpenAI text-embedding-ada-002
        elif self.model_type == "upstage":
            return 4096  # 업스테이지 solar-embedding-1-large-query
        else:
            # 테스트 임베딩으로 차원 확인
            try:
                test_embedding = self.embed_query("test")
                return len(test_embedding)
            except:
                return 768  # 기본 BERT 차원
    
    def get_model_info(self) -> dict:
        """현재 사용 중인 모델 정보 반환"""
        model_name = "unknown"
        if self.model_type == "upstage":
            model_name = settings.upstage_embedding_model
        else:
            model_name = getattr(self.embeddings, 'model_name', 'unknown')
        
        return {
            "type": self.model_type,
            "model_name": model_name,
            "dimension": self.get_embedding_dimension(),
            "is_korean_optimized": self.model_type == "huggingface" and any(
                korean_model in str(getattr(self.embeddings, 'model_name', ''))
                for korean_model in self.korean_models.values()
            )
        }