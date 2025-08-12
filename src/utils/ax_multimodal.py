"""
SKT A.X 4.0 VL Light 멀티모달 모델 - 이미지 분석 및 주제 관련성 판단

Hugging Face: skt/A.X-4.0-VL-Light

참고: https://huggingface.co/skt/A.X-4.0-VL-Light
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)


logger = logging.getLogger(__name__)


class AXMultimodalModel:
    """SKT A.X 4.0 VL Light를 사용한 멀티모달 이미지 분석"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_memory_gb: int = 8,
    ):
        self.model_path = self._get_model_path(model_path)
        self.device = self._setup_device(device)
        self.max_memory_gb = max_memory_gb

        self.model = None
        self.processor = None
        self._load_model()

    def _get_model_path(self, model_path: Optional[str] = None) -> Path:
        if model_path:
            return Path(model_path)

        # 기본 경로: model_bootstrap 우선
        try:
            from src.utils.model_bootstrap import get_ax_vl_dir
            d = get_ax_vl_dir()
            if d.exists() and (d / "config.json").exists():
                return d
        except Exception:
            pass

        # 환경 변수 또는 로컬 기본
        local_root = Path(os.getenv("LOCAL_MODELS_DIR", "/local_models"))
        candidate = local_root / "multimodal" / "A.X-4.0-VL-Light"
        if candidate.exists():
            return candidate

        # 프로젝트 models 폴더 폴백
        project_root = Path(__file__).parent.parent.parent
        models_candidate = project_root / "models" / "multimodal" / "A.X-4.0-VL-Light"
        if models_candidate.exists():
            return models_candidate

        # HuggingFace 모델 ID 폴백
        return Path("skt/A.X-4.0-VL-Light")

    def _setup_device(self, device: str) -> str:
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
        try:
            model_id = str(self.model_path)
            p = Path(model_id)
            if not p.exists():
                logger.info(f"🔄 HuggingFace에서 모델 로드: {model_id}")
            else:
                logger.info(f"🔄 로컬 모델 로드: {model_id}")

            # Processor
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            # dtype/장치 설정
            if self.device == "cuda":
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32

            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                "use_safetensors": True,
            }
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {0: f"{self.max_memory_gb}GB", "cpu": f"{self.max_memory_gb * 2}GB"}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
            )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("✅ A.X 멀티모달 모델 로드 완료!")

        except Exception as e:
            logger.error(f"❌ A.X 모델 로드 실패: {e}")
            self.model = None
            self.processor = None

    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def _build_conversations(self, prompt: str) -> List[Dict[str, Any]]:
        # A.X 모델 카드의 대화 포맷에 맞춰 구성
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def analyze_image(
        self,
        image_path: str,
        document_topic: Optional[Dict[str, Any]] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        if not self.model or not self.processor:
            logger.error("A.X 모델이 로드되지 않았습니다.")
            return {"error": "Model not loaded"}

        try:
            image = Image.open(image_path).convert("RGB")
            image = self._resize_image(image)

            prompt_parts = ["이미지를 분석하여 한국어로 답변하세요.\n"]
            if document_topic:
                prompt_parts.append(f"문서 주제: {document_topic.get('main_topic', 'Unknown')}\n")
                if document_topic.get("keywords"):
                    prompt_parts.append(f"키워드: {', '.join(document_topic['keywords'][:5])}\n")
            if context:
                prompt_parts.append(f"컨텍스트: {context[:200]}\n")
            prompt_parts.append(
                """
아래 형식으로 간결히 제공:
1) 유형(text/diagram/photo/chart)
2) 주요 내용 요약
3) 주제 관련도(0-1)
4) 텍스트 전용 여부(yes/no)
"""
            )
            prompt = "".join(prompt_parts)

            # 우선 대화 포맷 시도
            inputs = None
            try:
                conversations = self._build_conversations(prompt)
                inputs = self.processor(
                    images=[image],
                    conversations=[conversations],
                    padding=True,
                    return_tensors="pt",
                )
            except Exception:
                # 폴백: 일반 텍스트/이미지 입력
                inputs = self.processor(
                    images=[image],
                    text=[prompt],
                    padding=True,
                    return_tensors="pt",
                )

            if self.device != "cpu":
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                )

            try:
                # 일부 프로세서는 batch_decode만 지원
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            except Exception:
                response = self.processor.decode(outputs[0], skip_special_tokens=True)

            if prompt in response:
                response = response.replace(prompt, "").strip()

            return self._parse_response(response, document_topic)

        except Exception as e:
            logger.error(f"이미지 분석 실패(A.X): {e}")
            return {"error": str(e)}

    def _parse_response(self, response: str, document_topic: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        result = {
            "image_type": "unknown",
            "content_description": "",
            "relevance_score": 0.5,
            "is_text_only": False,
            "raw_response": response,
        }

        text = response.lower()
        if "text-only" in text:
            result["is_text_only"] = True

        # 간단 파싱
        for line in response.split("\n"):
            s = line.strip()
            if not s:
                continue
            if "type:" in s.lower():
                t = s.split(":", 1)[1].strip().lower()
                if "text" in t:
                    result["image_type"] = "text"
                    result["is_text_only"] = True
                elif "diagram" in t or "도표" in t:
                    result["image_type"] = "diagram"
                elif "photo" in t or "사진" in t:
                    result["image_type"] = "photo"
                elif "chart" in t or "그래프" in t:
                    result["image_type"] = "chart"
            elif "relevance:" in s.lower() or "관련도" in s:
                import re
                nums = re.findall(r"[\d.]+", s)
                if nums:
                    score = float(nums[0])
                    result["relevance_score"] = min(max(score, 0.0), 1.0)
            elif "content:" in s.lower() or "내용" in s:
                result["content_description"] = s.split(":", 1)[1].strip()

        if not result["content_description"] and response:
            for line in response.split("\n"):
                if len(line.strip()) > 10:
                    result["content_description"] = line.strip()
                    break

        return result

    def classify_image(self, image_path: str) -> Tuple[bool, float]:
        if not self.model or not self.processor:
            return False, 0.5
        try:
            image = Image.open(image_path).convert("RGB")
            image = self._resize_image(image, max_size=512)
            prompt = (
                "이 이미지는 주로 텍스트인가요, 아니면 의미있는 이미지인가요?\n"
                "답변: text-only 또는 meaningful-image\n"
                "신뢰도: 0-1"
            )
            conversations = self._build_conversations(prompt)
            try:
                inputs = self.processor(
                    images=[image],
                    conversations=[conversations],
                    padding=True,
                    return_tensors="pt",
                )
            except Exception:
                inputs = self.processor(
                    images=[image],
                    text=[prompt],
                    padding=True,
                    return_tensors="pt",
                )
            if self.device != "cpu":
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.1,
                    do_sample=False,
                )
            try:
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            except Exception:
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
            text = response.lower()
            is_text_only = ("text-only" in text) or ("text only" in text)
            confidence = 0.8
            import re
            nums = re.findall(r"[\d.]+", text)
            if nums:
                confidence = float(nums[0])
                if confidence > 1:
                    confidence = confidence / 100
            return is_text_only, confidence
        except Exception as e:
            logger.error(f"이미지 분류 실패(A.X): {e}")
            return False, 0.5

    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("A.X 멀티모달 모델 메모리 해제 완료")


def create_ax_multimodal_model(**kwargs) -> AXMultimodalModel:
    return AXMultimodalModel(**kwargs)


