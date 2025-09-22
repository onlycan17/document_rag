"""
문서 전처리 모델 인터페이스 및 구현체

PDF 문서의 텍스트 추출 및 전처리를 위한 로컬 및 외부 API 모델을 제공합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from config import settings

from .document_pipeline import build_metadata, concatenate_documents, load_documents

logger = logging.getLogger(__name__)


class PreprocessingModel(ABC):
    """문서 전처리 모델의 추상 기본 클래스"""
    
    def __init__(self, model_name: str = "default"):
        """
        전처리 모델 초기화
        
        Args:
            model_name: 사용할 모델 이름
        """
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def preprocess_text(self, text: str, **kwargs) -> str:
        """
        텍스트를 전처리합니다.
        
        Args:
            text: 전처리할 텍스트
            **kwargs: 추가 매개변수
            
        Returns:
            전처리된 텍스트
        """
        pass
    
    @abstractmethod
    def extract_and_preprocess(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        파일에서 텍스트를 추출하고 전처리합니다.
        
        Args:
            file_path: 처리할 파일 경로
            **kwargs: 추가 매개변수
            
        Returns:
            전처리 결과를 포함한 딕셔너리
            - text: 전처리된 텍스트
            - metadata: 메타데이터
            - processing_method: 사용된 처리 방법
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """모델 사용 가능 여부를 확인합니다."""
        pass
    
    def get_model_info(self) -> Dict[str, str]:
        """모델 정보를 반환합니다."""
        return {
            "model_type": self.__class__.__name__,
            "model_name": self.model_name,
            "description": self.__doc__ or "문서 전처리 모델"
        }


class LocalPreprocessingModel(PreprocessingModel):
    """로컬 모델을 사용한 문서 전처리"""
    
    def __init__(self, model_name: str = "local_default"):
        """
        로컬 전처리 모델 초기화
        
        Args:
            model_name: 로컬 모델 이름
        """
        super().__init__(model_name)
        self._initialized = False
    
    def _initialize_model(self):
        """로컬 모델을 초기화합니다."""
        if self._initialized:
            return
            
        try:
            # 로컬 모델 초기화 로직
            self.logger.info(f"로컬 전처리 모델 초기화: {self.model_name}")
            self._initialized = True
        except Exception as e:
            self.logger.error(f"로컬 모델 초기화 실패: {e}")
            raise
    
    def preprocess_text(self, text: str, **kwargs) -> str:
        """
        로컬 모델을 사용하여 텍스트를 전처리합니다.
        
        Args:
            text: 전처리할 텍스트
            **kwargs: 추가 매개변수
            
        Returns:
            전처리된 텍스트
        """
        self._initialize_model()
        
        try:
            # 기본적인 텍스트 정리
            processed_text = text.strip()
            
            # 한국어 문장 연결 개선
            processed_text = self._improve_korean_text_connection(processed_text)
            
            # 불필요한 공백 제거
            processed_text = self._remove_excessive_whitespace(processed_text)
            
            self.logger.info(f"로컬 전처리 완료: {len(text)} -> {len(processed_text)} 문자")
            return processed_text
            
        except Exception as e:
            self.logger.error(f"로컬 텍스트 전처리 실패: {e}")
            return text  # 실패 시 원본 반환
    
    def extract_and_preprocess(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        파일에서 텍스트를 추출하고 로컬 모델로 전처리합니다.
        
        Args:
            file_path: 처리할 파일 경로
            **kwargs: 추가 매개변수
            
        Returns:
            전처리 결과 딕셔너리
        """
        self._initialize_model()
        
        try:
            result = load_documents(
                file_path,
                use_ocr=kwargs.get('use_ocr', True),
                use_agent_preprocessing=False,
                enable_postprocessing=kwargs.get('enable_postprocessing', True),
            )
            if not result.success:
                return {
                    "text": "",
                    "metadata": {"error": result.error or "문서를 추출할 수 없습니다"},
                    "processing_method": "local_extraction_failed",
                }
            extracted_text = concatenate_documents(result.documents)
            processed_text = self.preprocess_text(extracted_text, **kwargs)
            metadata = build_metadata(
                original_length=len(extracted_text),
                processed_length=len(processed_text),
                document_count=len(result.documents),
                file_path=file_path,
                extra={"processing_method": "local_preprocessing"},
            )
            return {
                "text": processed_text,
                "metadata": metadata,
                "processing_method": "local_preprocessing",
            }
        except Exception as e:
            self.logger.error(f"로컬 파일 전처리 실패: {e}")
            return {
                "text": "",
                "metadata": {"error": str(e)},
                "processing_method": "local_preprocessing_failed",
            }
    
    def is_available(self) -> bool:
        """로컬 모델 사용 가능 여부를 확인합니다."""
        try:
            # 로컬 모델 사용 가능성 확인
            return True  # 기본적으로 로컬 모델은 항상 사용 가능
        except Exception:
            return False
    
    def _improve_korean_text_connection(self, text: str) -> str:
        """한국어 텍스트의 문장 연결을 개선합니다."""
        # 기본적인 한국어 문장 연결 개선
        # 실제로는 더 정교한 로컬 모델을 사용할 수 있습니다
        improved = text.replace(".\n", ". ")
        improved = improved.replace("?\n", "? ")
        improved = improved.replace("!\n", "! ")
        return improved
    
    def _remove_excessive_whitespace(self, text: str) -> str:
        """과도한 공백을 제거합니다."""
        import re
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 줄바꿈 정리
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        return text.strip()


class APIPreprocessingModel(PreprocessingModel):
    """외부 API 모델을 사용한 문서 전처리"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", provider: str = "openai"):
        """
        API 전처리 모델 초기화
        
        Args:
            model_name: 사용할 API 모델 이름
            provider: API 제공자 (openai, google, anthropic)
        """
        super().__init__(model_name)
        self.provider = provider
        self._client = None
        self._initialized = False
    
    def _initialize_client(self):
        """API 클라이언트를 초기화합니다."""
        if self._initialized:
            return
            
        try:
            from config import settings
            
            if self.provider == "openai":
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API 키가 설정되지 않았습니다")
                import openai
                self._client = openai.OpenAI(api_key=settings.openai_api_key)
                
            elif self.provider == "google":
                if not settings.google_api_key:
                    raise ValueError("Google API 키가 설정되지 않았습니다")
                import google.generativeai as genai
                genai.configure(api_key=settings.google_api_key)
                self._client = genai.GenerativeModel(self.model_name)
                
            elif self.provider == "anthropic":
                if not settings.anthropic_api_key:
                    raise ValueError("Anthropic API 키가 설정되지 않았습니다")
                import anthropic
                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                
            else:
                raise ValueError(f"지원하지 않는 제공자: {self.provider}")
            
            self._initialized = True
            self.logger.info(f"API 전처리 모델 초기화 완료: {self.provider} - {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"API 클라이언트 초기화 실패: {e}")
            raise
    
    def preprocess_text(self, text: str, **kwargs) -> str:
        """
        API 모델을 사용하여 텍스트를 전처리합니다.
        
        Args:
            text: 전처리할 텍스트
            **kwargs: 추가 매개변수
            
        Returns:
            전처리된 텍스트
        """
        self._initialize_client()
        
        try:
            # 전처리 프롬프트
            preprocessing_prompt = f"""
            다음 텍스트를 전문적으로 전처리해주세요:

            1. 한국어 문장의 연결성을 개선하세요
            2. 적절한 띄어쓰기를 적용하세요  
            3. 문맥에 맞는 문장 구조를 만드세요
            4. 불필요한 공백과 줄바꿈을 정리하세요
            5. 전문적이고 읽기 쉬운 텍스트로 만드세요

            텍스트:
            {text}

            전처리된 텍스트만 반환해주세요.
            """
            
            if self.provider == "openai":
                processed_text = None
                # 최신 모델(o3/o4/gpt-5/4.1 등) 호환: Responses API 우선 시도, 실패 시 Chat Completions 폴백
                try:
                    if hasattr(self._client, "responses"):
                        resp = self._client.responses.create(
                            model=self.model_name,
                            input=preprocessing_prompt,
                            temperature=settings.preprocessing_temperature,
                            max_output_tokens=4000,
                        )
                        # openai>=1.0.0 에서 제공되는 편의 프로퍼티 시도
                        processed_text = getattr(resp, "output_text", None)
                        if not processed_text:
                            # 구조적 필드 폴백 (버전/형식 차이 대응)
                            try:
                                processed_text = resp.output[0].content[0].text  # type: ignore[attr-defined]
                            except Exception:
                                processed_text = None
                except Exception:
                    processed_text = None

                if not processed_text:
                    # Chat Completions 폴백 (gpt-4o/4o-mini 등 호환)
                    response = self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": preprocessing_prompt}],
                        max_tokens=4000,
                        temperature=0.3,
                    )
                    processed_text = response.choices[0].message.content
                
            elif self.provider == "google":
                response = self._client.generate_content(preprocessing_prompt)
                processed_text = response.text
                
            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": preprocessing_prompt}]
                )
                processed_text = response.content[0].text
                
            else:
                raise ValueError(f"지원하지 않는 제공자: {self.provider}")
            
            self.logger.info(f"API 전처리 완료: {self.provider} - {len(text)} -> {len(processed_text)} 문자")
            return processed_text.strip()
            
        except Exception as e:
            self.logger.error(f"API 텍스트 전처리 실패: {e}")
            return text  # 실패 시 원본 반환
    
    def extract_and_preprocess(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        파일에서 텍스트를 추출하고 API 모델로 전처리합니다.
        
        Args:
            file_path: 처리할 파일 경로
            **kwargs: 추가 매개변수
            
        Returns:
            전처리 결과 딕셔너리
        """
        self._initialize_client()
        
        try:
            result = load_documents(
                file_path,
                use_ocr=kwargs.get('use_ocr', True),
                use_agent_preprocessing=False,
                enable_postprocessing=False,
            )
            if not result.success:
                return {
                    "text": "",
                    "metadata": {"error": result.error or "문서를 추출할 수 없습니다"},
                    "processing_method": f"{self.provider}_extraction_failed",
                }
            extracted_text = concatenate_documents(result.documents)
            processed_text = self.preprocess_text(extracted_text, **kwargs)
            metadata = build_metadata(
                original_length=len(extracted_text),
                processed_length=len(processed_text),
                document_count=len(result.documents),
                file_path=file_path,
                extra={
                    "api_provider": self.provider,
                    "api_model": self.model_name,
                    "processing_method": f"{self.provider}_api_preprocessing",
                },
            )
            return {
                "text": processed_text,
                "metadata": metadata,
                "processing_method": f"{self.provider}_api_preprocessing",
            }
        except Exception as e:
            self.logger.error(f"API 파일 전처리 실패: {e}")
            return {
                "text": "",
                "metadata": {"error": str(e)},
                "processing_method": f"{self.provider}_api_preprocessing_failed",
            }
    
    def is_available(self) -> bool:
        """API 모델 사용 가능 여부를 확인합니다."""
        try:
            self._initialize_client()
            return True
        except Exception as e:
            self.logger.warning(f"API 모델 사용 불가: {e}")
            return False
