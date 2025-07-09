from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
        else:
            # 한국어 성능이 우수한 모델 우선 선택
            model_name = self._select_best_korean_model()
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,  # 배치 크기 최적화
                    'show_progress_bar': False
                }
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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서를 배치로 임베딩
        - 빈 텍스트 필터링
        - 배치 처리 최적화
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
        except TypeError as e:
            # show_progress_bar 매개변수 충돌 처리
            if "show_progress_bar" in str(e) and self.model_type == "huggingface":
                # HuggingFace 모델의 경우 직접 encode 메서드 호출
                model = self.embeddings.client
                embeddings = model.encode(filtered_texts, show_progress_bar=False)
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
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리를 임베딩
        - 쿼리 전처리 및 확장
        """
        # 쿼리 전처리
        processed_query = self._preprocess_query(text)
        
        if not processed_query:
            return [0.0] * self.get_embedding_dimension()
        
        try:
            return self.embeddings.embed_query(processed_query)
        except TypeError as e:
            # show_progress_bar 매개변수 충돌 처리
            if "show_progress_bar" in str(e) and self.model_type == "huggingface":
                # HuggingFace 모델의 경우 직접 encode 메서드 호출
                model = self.embeddings.client
                embeddings = model.encode([processed_query], show_progress_bar=False)
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
        if not text or not isinstance(text, str):
            return ""
        
        # 공백 정규화
        text = ' '.join(text.split())
        
        # 특수 문자 정리
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # BOM
        text = text.replace('\xa0', ' ')   # Non-breaking space
        
        # 너무 짧은 텍스트 제외
        if len(text.strip()) < 10:
            return ""
        
        return text.strip()
    
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
        
        # 쿼리 확장 (선택적)
        # 예: "시스템"을 "정보시스템" 등으로 확장
        query_expansions = {
            "시스템": "정보시스템 시스템",
            "구축": "구축 건설 설치",
            "운영": "운영 관리 유지보수",
            "지침": "지침 가이드 규정"
        }
        
        for original, expanded in query_expansions.items():
            if original in query and original != query:
                query = query.replace(original, expanded)
        
        return query
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        if self.model_type == "openai":
            return 1536  # OpenAI text-embedding-ada-002
        else:
            # 테스트 임베딩으로 차원 확인
            try:
                test_embedding = self.embed_query("test")
                return len(test_embedding)
            except:
                return 768  # 기본 BERT 차원
    
    def get_model_info(self) -> dict:
        """현재 사용 중인 모델 정보 반환"""
        return {
            "type": self.model_type,
            "model_name": getattr(self.embeddings, 'model_name', 'unknown'),
            "dimension": self.get_embedding_dimension(),
            "is_korean_optimized": self.model_type == "huggingface" and any(
                korean_model in str(getattr(self.embeddings, 'model_name', ''))
                for korean_model in self.korean_models.values()
            )
        }