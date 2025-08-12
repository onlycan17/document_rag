"""
이미지 분석기 - Vision 모델을 활용한 이미지 설명 생성
"""

import os
import base64
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image
import requests

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    Vision 모델을 사용하여 이미지 분석 및 설명 생성
    """
    
    def __init__(self, model_provider: str = "openai"):
        """
        이미지 분석기 초기화
        
        Args:
            model_provider: 사용할 모델 제공업체 ('openai', 'google', 'anthropic')
        """
        self.model_provider = model_provider.lower()
        self.api_key = self._get_api_key()
        self.local_model = None  # Gemma 멀티모달 모델 (로컬)
        
        if self.model_provider == "local":
            # 로컬 멀티모달 모델(A.X) 초기화 시도
            try:
                from src.utils.model_bootstrap import get_ax_vl_dir
                from src.utils.ax_multimodal import AXMultimodalModel
                ax_dir = get_ax_vl_dir()
                if ax_dir and ax_dir.exists():
                    self.local_model = AXMultimodalModel(
                        model_path=str(ax_dir),
                        device="auto",
                        max_memory_gb=8,
                    )
                    logger.info("🧠 로컬 A.X 4.0 VL Light 모델 활성화됨 (이미지 설명용)")
                else:
                    logger.warning("로컬 A.X 4.0 VL Light 디렉토리를 찾을 수 없어 이미지 분석 로컬 모드를 비활성화합니다.")
            except Exception as e:
                logger.warning(f"로컬 멀티모달 모델 초기화 실패: {e}")
        else:
            if not self.api_key:
                logger.warning(f"⚠️ {model_provider} API 키가 설정되지 않았습니다. 이미지 분석 기능이 비활성화됩니다.")
        
        logger.info(f"🤖 이미지 분석기 초기화 완료 - 모델: {self.model_provider}")
    
    def _get_api_key(self) -> Optional[str]:
        """API 키 가져오기"""
        if self.model_provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.model_provider == "google":
            return os.getenv("GOOGLE_API_KEY")
        elif self.model_provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """이미지를 Base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _optimize_image_for_analysis(self, image_path: str) -> str:
        """분석을 위해 이미지 최적화 (크기 조정, 포맷 변환)"""
        try:
            with Image.open(image_path) as img:
                # 이미지 크기 확인 및 조정 (최대 2048x2048)
                max_size = 2048
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # RGB로 변환 (RGBA나 다른 모드인 경우)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 임시 파일로 저장
                temp_path = str(Path(image_path).with_suffix('.optimized.jpg'))
                img.save(temp_path, 'JPEG', quality=85, optimize=True)
                
                return temp_path
                
        except Exception as e:
            logger.warning(f"이미지 최적화 실패: {e}")
            return image_path
    
    def analyze_image_openai(self, image_path: str, context: str = "") -> Optional[str]:
        """OpenAI GPT-4V를 사용한 이미지 분석"""
        try:
            # 이미지 최적화
            optimized_path = self._optimize_image_for_analysis(image_path)
            base64_image = self._encode_image_to_base64(optimized_path)
            
            # API 요청 준비
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 프롬프트 구성
            system_prompt = """당신은 학술 문서의 이미지를 분석하는 전문가입니다. 
이미지를 자세히 관찰하고 다음 요소들을 포함한 한국어 설명을 제공해주세요:

1. 이미지 유형 (도표, 그래프, 사진, 도면, 지도 등)
2. 주요 내용과 데이터
3. 중요한 세부사항 (숫자, 라벨, 범례 등)
4. 학술적 맥락에서의 의미

설명은 150-300자 내외로 작성하며, 객관적이고 정확한 정보만 포함해주세요."""

            user_prompt = f"다음 이미지를 분석해주세요."
            if context:
                user_prompt += f"\n\n문맥 정보: {context}"

            payload = {
                "model": "gpt-4o",  # GPT-4 Vision
                "messages": [
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            # API 호출
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result['choices'][0]['message']['content'].strip()
                
                # 임시 파일 정리
                if optimized_path != image_path:
                    try:
                        os.remove(optimized_path)
                    except:
                        pass
                
                return description
            else:
                logger.error(f"OpenAI API 오류: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI 이미지 분석 실패: {str(e)}")
            return None

    def analyze_image_local(self, image_path: str, context: str = "") -> Optional[str]:
        """로컬 Gemma 멀티모달 모델을 사용한 이미지 분석"""
        try:
            if self.local_model is None:
                logger.warning("로컬 멀티모달 모델이 초기화되지 않았습니다.")
                return None

            # 문맥 정보를 document_topic으로 최소 전달
            document_topic = {
                "main_topic": "",
                "keywords": []
            }

            analysis = self.local_model.analyze_image(
                image_path,
                document_topic,
                context or ""
            )

            description = (analysis or {}).get("content_description", "").strip()
            if description:
                logger.info(f"✅ 이미지 분석 완료: {len(description)}자")
                return description
            else:
                logger.warning("이미지 분석 결과를 받지 못했습니다.")
                return None

        except Exception as e:
            logger.error(f"로컬 이미지 분석 실패: {str(e)}")
            return None
    
    def analyze_image_google(self, image_path: str, context: str = "") -> Optional[str]:
        """Google Gemini Vision을 사용한 이미지 분석"""
        try:
            # 동적 임포트로 린터/환경 의존성 문제 회피
            genai = __import__("google.generativeai", fromlist=["generativeai"])

            # API 키 설정
            genai.configure(api_key=self.api_key)

            # 모델 초기화
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # 이미지 로드
            img = Image.open(image_path)
            
            # 프롬프트 구성
            prompt = """이 이미지를 한국어로 자세히 분석해주세요. 다음을 포함해주세요:

1. 이미지 유형 (도표, 그래프, 사진, 도면 등)
2. 주요 내용과 데이터
3. 중요한 텍스트나 수치
4. 학술적 의미

150-300자 내외로 객관적이고 정확하게 설명해주세요."""
            
            if context:
                prompt += f"\n\n문맥 정보: {context}"
            
            # 이미지 분석 요청
            response = model.generate_content([prompt, img])
            
            if response.text:
                return response.text.strip()
            else:
                logger.error("Google Gemini에서 응답을 받지 못했습니다.")
                return None
                
        except Exception as e:
            logger.error(f"Google Gemini 이미지 분석 실패: {str(e)}")
            return None
    
    def analyze_image(self, image_path: str, context: str = "") -> Optional[str]:
        """
        이미지 분석 및 설명 생성
        
        Args:
            image_path: 분석할 이미지 파일 경로
            context: 추가 컨텍스트 정보 (페이지 내용 등)
            
        Returns:
            이미지 설명 텍스트 또는 None
        """
        if self.model_provider != "local":
            if not self.api_key:
                logger.warning("API 키가 없어 이미지 분석을 건너뜁니다.")
                return None
        
        if not os.path.exists(image_path):
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
        
        logger.info(f"🖼️ 이미지 분석 시작: {os.path.basename(image_path)}")
        
        try:
            # 모델별 분석 수행
            if self.model_provider == "openai":
                description = self.analyze_image_openai(image_path, context)
            elif self.model_provider == "google":
                description = self.analyze_image_google(image_path, context)
            elif self.model_provider == "local":
                description = self.analyze_image_local(image_path, context)
            else:
                logger.error(f"지원하지 않는 모델 제공업체: {self.model_provider}")
                return None
            
            if description:
                logger.info(f"✅ 이미지 분석 완료: {len(description)}자")
                return description
            else:
                logger.warning("이미지 분석 결과를 받지 못했습니다.")
                return None
                
        except Exception as e:
            logger.error(f"이미지 분석 중 오류 발생: {str(e)}")
            return None
    
    def get_image_type(self, image_path: str) -> str:
        """이미지 파일 타입 확인"""
        try:
            with Image.open(image_path) as img:
                return img.format.lower() if img.format else "unknown"
        except:
            return "unknown"
    
    def batch_analyze_images(self, image_paths: List[str], contexts: List[str] = None) -> Dict[str, Optional[str]]:
        """
        여러 이미지를 배치로 분석
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            contexts: 각 이미지에 대한 컨텍스트 리스트 (선택적)
            
        Returns:
            {이미지_경로: 설명} 딕셔너리
        """
        results = {}
        contexts = contexts or [""] * len(image_paths)
        
        logger.info(f"📦 배치 이미지 분석 시작: {len(image_paths)}개 이미지")
        
        for i, (image_path, context) in enumerate(zip(image_paths, contexts), 1):
            logger.info(f"🔄 진행 상황: {i}/{len(image_paths)} - {os.path.basename(image_path)}")
            
            try:
                description = self.analyze_image(image_path, context)
                results[image_path] = description
            except Exception as e:
                logger.error(f"❌ 이미지 분석 실패: {image_path} - {str(e)}")
                results[image_path] = None
        
        successful_count = sum(1 for desc in results.values() if desc is not None)
        logger.info(f"📦 배치 이미지 분석 완료: {successful_count}/{len(image_paths)}개 성공")
        
        return results


def create_image_analyzer(model_provider: str = None) -> ImageAnalyzer:
    """
    환경 변수를 고려하여 이미지 분석기 생성
    
    Args:
        model_provider: 사용할 모델 제공업체 (None이면 환경 변수 확인)
        
    Returns:
        ImageAnalyzer 인스턴스
    """
    if model_provider is None:
        # 로컬 Gemma가 있으면 최우선 사용
        try:
            from src.utils.model_bootstrap import get_gemma_dir
            gemma_dir = get_gemma_dir(prefer_3n=True)
            if gemma_dir and gemma_dir.exists():
                return ImageAnalyzer(model_provider="local")
        except Exception:
            pass

        # 환경 변수에서 사용 가능한 모델 확인
        if os.getenv("OPENAI_API_KEY"):
            model_provider = "openai"
        elif os.getenv("GOOGLE_API_KEY"):
            model_provider = "google"
        elif os.getenv("ANTHROPIC_API_KEY"):
            model_provider = "anthropic"
        else:
            # 기본은 로컬 시도 (사용자 요구: 로컬 우선)
            model_provider = "local"
    
    return ImageAnalyzer(model_provider=model_provider)


if __name__ == "__main__":
    # 테스트 코드
    analyzer = create_image_analyzer()
    
    # 테스트 이미지 분석
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        description = analyzer.analyze_image(test_image, "백제 토기에 관한 문서")
        print(f"이미지 설명: {description}")
    else:
        print("테스트 이미지가 없습니다.")