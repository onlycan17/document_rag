"""
Gemma 멀티모달 모델 - 이미지 분석 및 주제 관련성 판단

이 모듈은 Google의 Gemma-2-2b-it 모델을 사용하여
이미지를 분석하고 문서 주제와의 관련성을 판단합니다.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


class GemmaMultimodalModel:
    """Gemma-2를 사용한 멀티모달 이미지 분석"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 load_in_4bit: bool = True,
                 max_memory_gb: int = 8):
        """
        멀티모달 모델 초기화
        
        Args:
            model_path: 모델 디렉토리 경로
            device: 디바이스 설정 ("auto", "cuda", "cpu")
            load_in_4bit: 4비트 양자화 사용 여부
            max_memory_gb: 최대 메모리 사용량 (GB)
        """
        self.model_path = self._get_model_path(model_path)
        self.device = self._setup_device(device)
        self.load_in_4bit = load_in_4bit
        self.max_memory_gb = max_memory_gb
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _get_model_path(self, model_path: Optional[str] = None) -> Path:
        """모델 경로 가져오기"""
        if model_path:
            return Path(model_path)
        
        # 기본 경로
        project_root = Path(__file__).parent.parent.parent
        default_path = project_root / "models" / "multimodal" / "gemma-2-2b-it"
        
        # HuggingFace 모델 ID로 폴백
        if not default_path.exists():
            logger.info("로컬 모델이 없습니다. HuggingFace에서 직접 로드합니다.")
            return Path("google/gemma-2-2b-it")  # HF 모델 ID
            
        return default_path
    
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            if torch.cuda.is_available():
                logger.info("🎮 CUDA 사용 가능 - GPU 모드")
                return "cuda"
            elif torch.backends.mps.is_available():
                logger.info("🍎 MPS 사용 가능 - Apple Silicon")
                return "mps"
            else:
                logger.info("💻 CPU 모드")
                return "cpu"
        return device
    
    def _load_model(self):
        """모델과 프로세서 로드"""
        try:
            model_id = str(self.model_path)
            
            # 로컬 경로가 존재하지 않으면 HuggingFace ID 사용
            if not self.model_path.exists():
                model_id = "google/gemma-2-2b-it"
                logger.info(f"🔄 HuggingFace에서 모델 로드: {model_id}")
            else:
                logger.info(f"🔄 로컬 모델 로드: {model_id}")
            
            # 양자화 설정
            quantization_config = None
            if self.load_in_4bit and self.device != "cpu":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("   4비트 양자화 활성화")
            
            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # 모델 로드
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config
            
            if self.device != "cpu":
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {0: f"{self.max_memory_gb}GB", "cpu": f"{self.max_memory_gb * 2}GB"}
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                **load_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 평가 모드 설정
            self.model.eval()
            
            logger.info("✅ 멀티모달 모델 로드 완료!")
            logger.info(f"   디바이스: {self.device}")
            logger.info(f"   메모리 제한: {self.max_memory_gb}GB")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            logger.info("💡 팁: transformers>=4.53.0이 설치되어 있는지 확인하세요.")
            self.model = None
            self.processor = None
    
    def analyze_image(self, 
                     image_path: str,
                     document_topic: Optional[Dict[str, Any]] = None,
                     context: str = "") -> Dict[str, Any]:
        """
        이미지 분석 및 문서 관련성 판단
        
        Args:
            image_path: 이미지 파일 경로
            document_topic: 문서 주제 정보
            context: 이미지 주변 텍스트
            
        Returns:
            분석 결과 딕셔너리
        """
        if not self.model or not self.processor:
            logger.error("모델이 로드되지 않았습니다.")
            return {"error": "Model not loaded"}
        
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            
            # 이미지 크기 조정 (메모리 절약)
            image = self._resize_image(image)
            
            # 프롬프트 생성
            prompt = self._create_analysis_prompt(document_topic, context)
            
            # 입력 준비
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # 디바이스로 이동
            if self.device != "cpu":
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9
                )
            
            # 디코딩
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 제거
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # 결과 파싱
            return self._parse_response(response, document_topic)
            
        except Exception as e:
            logger.error(f"이미지 분석 실패: {e}")
            return {"error": str(e)}
    
    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """이미지 크기 조정"""
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    
    def _create_analysis_prompt(self, document_topic: Optional[Dict[str, Any]], context: str) -> str:
        """이미지 분석 프롬프트 생성"""
        prompt_parts = ["Analyze this image and answer in Korean.\n"]
        
        if document_topic:
            prompt_parts.append(f"Document topic: {document_topic.get('main_topic', 'Unknown')}\n")
            if document_topic.get('keywords'):
                prompt_parts.append(f"Keywords: {', '.join(document_topic['keywords'][:5])}\n")
        
        if context:
            prompt_parts.append(f"Context: {context[:200]}\n")
        
        prompt_parts.append("""
Please provide:
1. Image type (text/diagram/photo/chart)
2. Main content description
3. Relevance to document topic (0-1 score)
4. Whether this is text-only image (yes/no)

Format:
Type: [type]
Content: [description]
Relevance: [0-1 score]
Text-only: [yes/no]
""")
        
        return "".join(prompt_parts)
    
    def _parse_response(self, response: str, document_topic: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """응답 파싱"""
        result = {
            "image_type": "unknown",
            "content_description": "",
            "relevance_score": 0.5,
            "is_text_only": False,
            "raw_response": response
        }
        
        lines = response.lower().split('\n')
        for line in lines:
            line = line.strip()
            
            if 'type:' in line:
                type_text = line.split('type:', 1)[1].strip()
                if 'text' in type_text:
                    result["image_type"] = "text"
                    result["is_text_only"] = True
                elif 'diagram' in type_text or '도표' in type_text:
                    result["image_type"] = "diagram"
                elif 'photo' in type_text or '사진' in type_text:
                    result["image_type"] = "photo"
                elif 'chart' in type_text or '그래프' in type_text:
                    result["image_type"] = "chart"
            
            elif 'content:' in line:
                result["content_description"] = line.split('content:', 1)[1].strip()
            
            elif 'relevance:' in line:
                try:
                    score_text = line.split('relevance:', 1)[1].strip()
                    # 숫자 추출
                    import re
                    numbers = re.findall(r'[\d.]+', score_text)
                    if numbers:
                        score = float(numbers[0])
                        result["relevance_score"] = min(max(score, 0.0), 1.0)
                except:
                    pass
            
            elif 'text-only:' in line:
                result["is_text_only"] = 'yes' in line or '예' in line
        
        # 한글 응답 처리
        if not result["content_description"] and response:
            # 첫 번째 의미있는 문장을 설명으로 사용
            for line in response.split('\n'):
                if len(line.strip()) > 10:
                    result["content_description"] = line.strip()
                    break
        
        return result
    
    def classify_image(self, image_path: str) -> Tuple[bool, float]:
        """
        이미지를 텍스트 전용인지 실제 이미지인지 분류
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            (is_text_only, confidence)
        """
        if not self.model or not self.processor:
            return False, 0.5
        
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            image = self._resize_image(image, max_size=512)  # 작은 크기로
            
            # 간단한 분류 프롬프트
            prompt = """Is this image primarily text content that should be extracted as text, 
or is it a meaningful image (photo, diagram, chart) that should be preserved?
Answer: text-only or meaningful-image
Confidence: 0-1"""
            
            # 입력 준비
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False
                )
            
            # 디코딩
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip().lower()
            
            # 파싱
            is_text_only = 'text-only' in response or 'text only' in response
            
            # 신뢰도 추출
            confidence = 0.8  # 기본값
            import re
            numbers = re.findall(r'[\d.]+', response)
            if numbers:
                confidence = float(numbers[0])
                if confidence > 1:
                    confidence = confidence / 100
            
            return is_text_only, confidence
            
        except Exception as e:
            logger.error(f"이미지 분류 실패: {e}")
            return False, 0.5
    
    def cleanup(self):
        """모델 메모리 해제"""
        if self.model:
            del self.model
            self.model = None
        
        if self.processor:
            del self.processor
            self.processor = None
        
        # GPU 메모리 정리
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("멀티모달 모델 메모리 해제 완료")


# 편의 함수
def create_gemma_multimodal_model(**kwargs) -> GemmaMultimodalModel:
    """Gemma 멀티모달 모델 생성"""
    return GemmaMultimodalModel(**kwargs)


if __name__ == "__main__":
    # 테스트
    model = create_gemma_multimodal_model()
    
    # 테스트 이미지가 있다면
    test_image = "test_image.jpg"
    if Path(test_image).exists():
        # 문서 주제 (예시)
        doc_topic = {
            "main_topic": "몽촌토성",
            "keywords": ["백제", "토성", "한성", "발굴"],
            "domain": "역사"
        }
        
        # 이미지 분석
        result = model.analyze_image(test_image, doc_topic, "토성 발굴 현장")
        print("이미지 분석 결과:")
        print(result)
        
        # 분류
        is_text, confidence = model.classify_image(test_image)
        print(f"\n텍스트 전용 이미지: {is_text} (신뢰도: {confidence:.2f})")
    else:
        print("테스트 이미지가 없습니다.")
    
    # 정리
    model.cleanup()