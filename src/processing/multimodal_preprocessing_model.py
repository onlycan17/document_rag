"""
멀티모달 전처리 모델

이미지와 텍스트를 함께 처리할 수 있는 멀티모달 AI 모델을 사용한 문서 전처리
"""

from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import base64

from .preprocessing_model import APIPreprocessingModel
from config import settings

logger = logging.getLogger(__name__)


class MultimodalPreprocessingModel(APIPreprocessingModel):
    """멀티모달 AI 모델을 사용한 문서 전처리 (이미지 + 텍스트)"""
    
    SUPPORTED_MULTIMODAL_MODELS = {
        # OpenAI 멀티모달 지원 모델
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            # 공식 가격 페이지(미러) 확인: GPT-5 mini/nano
            # 참고: http(s)://openai.com/ko-KR/api/pricing/ (Cloudflare 우회 미러로 확인)
            "gpt-5-mini",
            "gpt-5-nano",
        ],
        # Google Gemini 멀티모달 지원 모델 (최신 문서 기준)
        # 참고: https://ai.google.dev/gemini-api/docs/models
        "google": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ],
        # Anthropic Claude 멀티모달 지원 모델 (비전 입력 지원 라인업)
        # 참고: https://docs.anthropic.com/en/docs/about-claude/models
        "anthropic": [
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
        ],
    }

    @classmethod
    def get_supported_multimodal_models(cls) -> Dict[str, list[str]]:
        """
        기본 지원 목록에 .env 확장 모델을 병합해 반환합니다.

        환경변수(콤마 구분):
        - EXTRA_MULTIMODAL_OPENAI_MODELS
        - EXTRA_MULTIMODAL_GOOGLE_MODELS
        - EXTRA_MULTIMODAL_ANTHROPIC_MODELS
        """
        base: Dict[str, list[str]] = {
            k: v.copy() for k, v in cls.SUPPORTED_MULTIMODAL_MODELS.items()
        }

        def extend_unique(dst: list[str], extra_csv: str | None) -> None:
            if not extra_csv:
                return
            for name in [s.strip() for s in extra_csv.split(",") if s.strip()]:
                if name not in dst:
                    dst.append(name)

        extend_unique(base.setdefault("openai", []), getattr(settings, "extra_multimodal_openai_models", None))
        extend_unique(base.setdefault("google", []), getattr(settings, "extra_multimodal_google_models", None))
        extend_unique(base.setdefault("anthropic", []), getattr(settings, "extra_multimodal_anthropic_models", None))

        return base
    
    def __init__(self, model_name: str, provider: str):
        """
        멀티모달 전처리 모델 초기화
        
        Args:
            model_name: 멀티모달 모델 이름
            provider: API 제공자
        """
        super().__init__(model_name, provider)
        self.is_multimodal = self._check_multimodal_support(model_name, provider)
        
        if not self.is_multimodal:
            logger.warning(f"{model_name}은(는) 멀티모달 모델이 아닙니다. 일반 텍스트 전처리 모드로 동작합니다.")
    
    def _check_multimodal_support(self, model_name: str, provider: str) -> bool:
        """모델이 멀티모달을 지원하는지 확인합니다."""
        supported = self.get_supported_multimodal_models()
        supported_models = supported.get(provider, [])
        return model_name in supported_models
    
    def preprocess_document_with_images(self, text: str, images: List[Dict[str, Any]], **kwargs) -> str:
        """
        텍스트와 이미지를 함께 전처리합니다.
        
        Args:
            text: 전처리할 텍스트
            images: 이미지 정보 리스트
            **kwargs: 추가 매개변수
            
        Returns:
            전처리된 텍스트
        """
        if not self.is_multimodal:
            logger.info("멀티모달을 지원하지 않는 모델입니다. 일반 텍스트 전처리를 수행합니다.")
            return self.preprocess_text(text, **kwargs)
        
        self._initialize_client()
        
        try:
            # 멀티모달 프롬프트 생성
            multimodal_prompt = self._create_multimodal_prompt(text, images, **kwargs)
            
            if self.provider == "openai":
                return self._process_openai_multimodal(multimodal_prompt, images)
            elif self.provider == "google":
                return self._process_google_multimodal(multimodal_prompt, images)
            elif self.provider == "anthropic":
                return self._process_anthropic_multimodal(multimodal_prompt, images)
            else:
                logger.warning(f"지원하지 않는 멀티모달 제공자: {self.provider}")
                return self.preprocess_text(text, **kwargs)
                
        except Exception as e:
            logger.error(f"멀티모달 전처리 실패: {e}. 일반 텍스트 전처리로 폴백합니다.")
            return self.preprocess_text(text, **kwargs)
    
    def _create_multimodal_prompt(self, text: str, images: List[Dict[str, Any]], **kwargs) -> str:
        """멀티모달 프롬프트를 생성합니다."""
        prompt = f"""
        다음 문서의 텍스트와 이미지를 함께 분석하여 전문적으로 전처리해주세요:

        **텍스트 내용:**
        {text}

        **작업 지침:**
        1. 이미지의 내용을 이해하고 텍스트와 관련성을 파악하세요
        2. 이미지에서 추출할 수 있는 텍스트 정보를 식별하세요
        3. 한국어 문장의 연결성을 개선하세요
        4. 적절한 띄어쓰기와 문장 구조를 적용하세요
        5. 이미지와 텍스트의 정보를 통합하여 완전한 문서를 만드세요
        6. 전문적이고 읽기 쉬운 텍스트로 정리하세요

        **이미지 정보:**
        총 {len(images)}개의 이미지가 있습니다.
        
        전처리된 텍스트만 반환해주세요.
        """
        return prompt
    
    def _process_openai_multimodal(self, prompt: str, images: List[Dict[str, Any]]) -> str:
        """OpenAI GPT-4V를 사용한 멀티모달 처리"""
        try:
            messages = [{"role": "user", "content": []}]
            
            # 텍스트 추가
            messages[0]["content"].append({"type": "text", "text": prompt})
            
            # 이미지 추가
            for image_info in images[:10]:  # 최대 10개 이미지 제한
                image_path = image_info.get('path', '')
                if os.path.exists(image_path):
                    # 이미지를 base64로 인코딩
                    with open(image_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode()
                    
                    # MIME 타입 결정
                    ext = Path(image_path).suffix.lower()
                    if ext in ['.jpg', '.jpeg']:
                        mime_type = 'image/jpeg'
                    elif ext == '.png':
                        mime_type = 'image/png'
                    elif ext == '.gif':
                        mime_type = 'image/gif'
                    else:
                        mime_type = 'image/png'
                    
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_string}",
                            "detail": "high"
                        }
                    })
            
            # API 호출
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.preprocessing_max_tokens,
                temperature=self.preprocessing_temperature
            )
            
            processed_text = response.choices[0].message.content
            logger.info(f"OpenAI 멀티모달 전처리 완료: {len(processed_text)}자")
            return processed_text.strip()
            
        except Exception as e:
            logger.error(f"OpenAI 멀티모달 처리 실패: {e}")
            raise
    
    def _process_google_multimodal(self, prompt: str, images: List[Dict[str, Any]]) -> str:
        """Google Gemini를 사용한 멀티모달 처리"""
        try:
            # Gemini용 콘텐츠 준비
            contents = [prompt]
            
            # 이미지 추가
            for image_info in images[:10]:  # 최대 10개 이미지 제한
                image_path = image_info.get('path', '')
                if os.path.exists(image_path):
                    from PIL import Image
                    image = Image.open(image_path)
                    contents.append(image)
            
            # API 호출
            response = self._client.generate_content(contents)
            processed_text = response.text
            
            logger.info(f"Google 멀티모달 전처리 완료: {len(processed_text)}자")
            return processed_text.strip()
            
        except Exception as e:
            logger.error(f"Google 멀티모달 처리 실패: {e}")
            raise
    
    def _process_anthropic_multimodal(self, prompt: str, images: List[Dict[str, Any]]) -> str:
        """Anthropic Claude를 사용한 멀티모달 처리"""
        try:
            messages = [{"role": "user", "content": []}]
            
            # 텍스트 추가
            messages[0]["content"].append({"type": "text", "text": prompt})
            
            # 이미지 추가
            for image_info in images[:10]:  # 최대 10개 이미지 제한
                image_path = image_info.get('path', '')
                if os.path.exists(image_path):
                    # 이미지를 base64로 인코딩
                    with open(image_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode()
                    
                    # MIME 타입 결정
                    ext = Path(image_path).suffix.lower()
                    if ext in ['.jpg', '.jpeg']:
                        media_type = 'image/jpeg'
                    elif ext == '.png':
                        media_type = 'image/png'
                    elif ext == '.gif':
                        media_type = 'image/gif'
                    else:
                        media_type = 'image/png'
                    
                    messages[0]["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_string
                        }
                    })
            
            # API 호출
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=self.preprocessing_max_tokens,
                messages=messages
            )
            
            processed_text = response.content[0].text
            logger.info(f"Anthropic 멀티모달 전처리 완료: {len(processed_text)}자")
            return processed_text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic 멀티모달 처리 실패: {e}")
            raise
    
    def extract_and_preprocess_with_images(self, file_path: str, images: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        파일에서 텍스트를 추출하고 이미지와 함께 전처리합니다.
        
        Args:
            file_path: 처리할 파일 경로
            images: 관련 이미지 리스트
            **kwargs: 추가 매개변수
            
        Returns:
            전처리 결과 딕셔너리
        """
        try:
            # 기본 텍스트 추출
            from src.loaders.document_loader import DocumentLoader
            
            loader = DocumentLoader(
                use_ocr=False,  # OCR은 멀티모달 모델이 처리
                use_agent_preprocessing=False,
                enable_postprocessing=False
            )
            
            documents = loader.load_document(file_path)
            
            if not documents:
                return {
                    "text": "",
                    "metadata": {"error": "문서를 추출할 수 없습니다"},
                    "processing_method": f"{self.provider}_multimodal_extraction_failed"
                }
            
            # 텍스트 추출
            extracted_text = "\n\n".join([doc.page_content for doc in documents])
            
            # 멀티모달 전처리
            processed_text = self.preprocess_document_with_images(extracted_text, images, **kwargs)
            
            # 메타데이터 수집
            metadata = {
                "original_length": len(extracted_text),
                "processed_length": len(processed_text),
                "document_count": len(documents),
                "file_path": file_path,
                "image_count": len(images),
                "api_provider": self.provider,
                "api_model": self.model_name,
                "processing_method": f"{self.provider}_multimodal_preprocessing",
                "multimodal_processing": True
            }
            
            return {
                "text": processed_text,
                "metadata": metadata,
                "processing_method": f"{self.provider}_multimodal_preprocessing"
            }
            
        except Exception as e:
            logger.error(f"멀티모달 파일 전처리 실패: {e}")
            return {
                "text": "",
                "metadata": {"error": str(e)},
                "processing_method": f"{self.provider}_multimodal_preprocessing_failed"
            }
    
    @property
    def preprocessing_max_tokens(self) -> int:
        """전처리 최대 토큰 수"""
        return settings.preprocessing_max_tokens
    
    @property
    def preprocessing_temperature(self) -> float:
        """전처리 온도"""
        return settings.preprocessing_temperature
