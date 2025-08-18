"""
Gemma 멀티모달 모델 - 이미지 분석 및 주제 관련성 판단

이 모듈은 Google의 Gemma-3n-e4b 모델을 우선 사용하여
이미지를 분석하고 문서 주제와의 관련성을 판단합니다.
로컬 경로가 없으면 HuggingFace의 `google/gemma-3n-e4b`를 사용합니다.
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
    BitsAndBytesConfig,
    AutoConfig,
)
try:  # transformers >= 4.52
    from transformers import AutoModelForImageTextToText as _AutoMMModel
except Exception:  # pragma: no cover - fallback for older versions
    try:
        from transformers import AutoModelForVision2Seq as _AutoMMModel
    except Exception:  # last resort: define placeholder to raise later
        _AutoMMModel = None  # type: ignore

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
        
        # 1. 환경변수 GEMMA_MULTIMODAL_DIR 최우선 확인
        env_dir = os.getenv("GEMMA_MULTIMODAL_DIR")
        if env_dir:
            env_path = Path(env_dir)
            if env_path.exists() and env_path.is_dir() and (env_path / "config.json").exists():
                logger.info(f"🎯 환경변수에서 모델 경로 사용: {env_path}")
                return env_path
        
        # 2. model_bootstrap을 통한 자동 탐색
        try:
            from src.utils.model_bootstrap import get_gemma_dir
            g = get_gemma_dir(prefer_3n=True)
            # 디렉토리 내에 config.json이 있으면 MLX 여부와 관계없이 허용 (Transformers 호환)
            if g.exists():
                if g.is_dir() and (g / "config.json").exists():
                    logger.info(f"🔍 model_bootstrap에서 모델 경로 발견: {g}")
                    return g
                # 파일(.gguf)은 멀티모달에 부적합 → 폴백
        except Exception as e:
            logger.warning(f"model_bootstrap 사용 실패: {e}")
        
        # 3. 기본 경로들 확인
        project_root = Path(__file__).parent.parent.parent
        local_root = Path(os.getenv("LOCAL_MODELS_DIR", "/local_models"))
        
        # 기본 경로 후보들
        default_paths = [
            local_root / "multimodal" / "gemma-3n-e4b",
            project_root / "models" / "multimodal" / "gemma-3n-e4b",
            local_root / "multimodal" / "A.X-4.0-VL-Light",
            project_root / "local_models" / "multimodal" / "A.X-4.0-VL-Light"
        ]
        
        for path in default_paths:
            if path.exists() and path.is_dir() and (path / "config.json").exists():
                logger.info(f"✅ 로컬 모델 경로 발견: {path}")
                return path
        
        # 4. HuggingFace 모델 ID로 폴백
        logger.info("로컬 모델이 없습니다. HuggingFace에서 직접 로드합니다.")
        return Path("google/gemma-3n-e4b")  # HF 모델 ID
    
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
    
    def _rebuild_index_if_needed(self, model_dir: Path) -> None:
        """로컬 샤드 파일 개수/이름과 index.json이 불일치하면 index를 재생성합니다.

        - safetensors의 메타데이터만 읽어 파라미터 키 목록을 수집합니다.
        - 각 키를 해당 샤드 파일로 매핑하여 weight_map을 구성합니다.
        - 총 파일 크기 합을 total_size로 기록합니다.
        """
        try:
            index_path = model_dir / "model.safetensors.index.json"
            shard_files = sorted([p for p in model_dir.glob("model-*-of-*.safetensors")])
            if not shard_files:
                return

            # 현재 index가 존재하고, 그 안의 샤드 이름과 실제 파일명이 모두 일치하면 종료
            if index_path.exists():
                try:
                    import json
                    data = json.loads(index_path.read_text(encoding="utf-8"))
                    mapped_files = set(data.get("weight_map", {}).values())
                    actual_files = set(f.name for f in shard_files)
                    # 매핑된 파일이 전부 실제에 포함되면 유지
                    if mapped_files and mapped_files.issubset(actual_files):
                        return
                except Exception:
                    pass

            # 재생성
            try:
                from safetensors import safe_open  # 메타만 읽기
            except Exception:
                logger.warning("safetensors가 설치되지 않아 index 재생성을 건너뜁니다")
                return

            weight_map: Dict[str, str] = {}
            for sf in shard_files:
                try:
                    with safe_open(str(sf), framework="pt") as f:
                        for key in f.keys():
                            weight_map[key] = sf.name
                except Exception as e:
                    logger.warning(f"샤드 키 스캔 실패({sf.name}): {e}")

            if not weight_map:
                return

            import json, os
            total_size = 0
            for sf in shard_files:
                try:
                    total_size += os.path.getsize(sf)
                except Exception:
                    pass

            rebuilt = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            index_path.write_text(json.dumps(rebuilt, ensure_ascii=False), encoding="utf-8")
            logger.info("   model.safetensors.index.json을 로컬 샤드에 맞게 재생성했습니다")
        except Exception as e:
            logger.warning(f"index 재생성 실패(무시): {e}")

    def _load_model(self):
        """모델과 프로세서 로드"""
        try:
            model_id = str(self.model_path)
            
            # 로컬 경로가 디렉토리/파일로 존재하지 않으면 HuggingFace ID 사용
            if not self.model_path.exists():
                model_id = "google/gemma-3n-e4b"
                logger.info(f"🔄 HuggingFace에서 모델 로드: {model_id}")
            else:
                logger.info(f"🔄 로컬 모델 로드: {model_id}")
            
            # GGUF 또는 비-Transformers 디렉토리 가드
            # GGUF 파일(.gguf)인 경우, 현재 구현은 Transformers 멀티모달 로더와 호환되지 않으므로 에러 로그 후 종료
            p = Path(model_id)
            if p.is_file() and p.suffix == ".gguf":
                logger.error("지정된 경로는 GGUF 파일입니다. 현재 멀티모달 로더는 Transformers 형식만 지원합니다.")
                self.model = None
                self.processor = None
                return
            if p.is_dir() and not (p / "config.json").exists():
                # Transformers 형식이 아니면 원격으로 폴백
                logger.warning("지정된 Gemma 경로에 config.json이 없습니다. HuggingFace 원격 모델로 폴백합니다.")
                model_id = "google/gemma-3n-e4b"

            # 양자화 설정
            quantization_config = None
            # bitsandbytes는 CUDA에서만 지원. MPS/CPU는 비활성화
            if self.load_in_4bit and self.device == "cuda":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("   4비트 양자화 활성화 (CUDA)")
                except Exception as qe:
                    logger.warning(f"bitsandbytes 구성이 불가하여 4bit 비활성화: {qe}")
                    quantization_config = None
            
            # 프로세서 로드 (trust_remote_code를 조건부로 설정)
            processor_kwargs = {"trust_remote_code": True}
            
            # A.X-4.0-VL-Light 모델의 경우 특별 처리
            if "A.X-4.0-VL-Light" in str(model_id):
                logger.info("🎯 A.X-4.0-VL-Light 전용 프로세서 로드")
                # 모델 디렉토리를 sys.path에 임시 추가
                import sys
                model_dir = str(Path(model_id).resolve())
                if model_dir not in sys.path:
                    sys.path.insert(0, model_dir)
                
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        model_id, 
                        **processor_kwargs
                    )
                finally:
                    if model_dir in sys.path:
                        sys.path.remove(model_dir)
            else:
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    **processor_kwargs
                )
            
            # 모델 로드
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            }
            
            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config
            
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {0: f"{self.max_memory_gb}GB", "cpu": f"{self.max_memory_gb * 2}GB"}
            
            if _AutoMMModel is None:
                raise RuntimeError("적합한 멀티모달 AutoModel 클래스를 찾지 못했습니다. transformers 버전을 업데이트 해주세요.")

            # 구성 불러와서 비호환 quantization 설정 제거 (MLX-4bit 등)
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            # 비호환 quantization 설정이 모델 구성에 저장된 경우 제거
            try:
                if hasattr(config, "quantization_config"):
                    try:
                        delattr(config, "quantization_config")
                        logger.info("   모델 config의 quantization_config 제거")
                    except Exception:
                        setattr(config, "quantization_config", None)
                        logger.info("   모델 config의 quantization_config를 None으로 설정")
            except Exception:
                pass

            # CPU/MPS에서 안전한 주의집중 구현
            if self.device != "cuda":
                load_kwargs["attn_implementation"] = "eager"

            try:
                # 모델 아키텍처 확인하여 적절한 로더 선택
                if hasattr(config, 'architectures') and config.architectures:
                    arch = config.architectures[0]
                    logger.info(f"🏗️ 감지된 모델 아키텍처: {arch}")
                    
                    if arch == "AX4VLForConditionalGeneration":
                        # A.X-4.0-VL-Light 모델용 사용자 정의 로더
                        logger.info("🎯 A.X-4.0-VL-Light 모델 로드 중...")
                        
                        # 모델 디렉토리를 Python 경로에 추가하여 사용자 정의 모듈 임포트 활성화
                        import sys
                        model_dir = str(Path(model_id).resolve())
                        if model_dir not in sys.path:
                            sys.path.insert(0, model_dir)
                        
                        try:
                            # 사용자 정의 모델 클래스 직접 임포트
                            from modeling_ax4vl import AX4VLForConditionalGeneration
                            logger.info("✅ 사용자 정의 모델 클래스 임포트 성공")
                            
                            self.model = AX4VLForConditionalGeneration.from_pretrained(
                                model_id,
                                config=config,
                                **load_kwargs
                            )
                        except ImportError as import_err:
                            logger.warning(f"사용자 정의 클래스 임포트 실패, AutoModel 사용: {import_err}")
                            from transformers import AutoModel
                            self.model = AutoModel.from_pretrained(
                                model_id,
                                config=config,
                                **load_kwargs
                            )
                        finally:
                            # Python 경로에서 모델 디렉토리 제거
                            if model_dir in sys.path:
                                sys.path.remove(model_dir)
                    else:
                        # 기본 Gemma 멀티모달 모델 로더
                        self.model = _AutoMMModel.from_pretrained(
                            model_id,
                            config=config,
                            **load_kwargs
                        )
                else:
                    # 아키텍처 정보가 없는 경우 기본 로더 사용
                    self.model = _AutoMMModel.from_pretrained(
                        model_id,
                        config=config,
                        **load_kwargs
                    )
            except (OSError, FileNotFoundError) as missing_err:
                # 샤드 불일치 가능성 → index 재생성 후 한 번 더 시도
                p = Path(model_id)
                if p.is_dir():
                    logger.warning(f"로컬 샤드 불일치 감지: {missing_err}. index 재생성 시도")
                    self._rebuild_index_if_needed(p)
                    self.model = _AutoMMModel.from_pretrained(
                        model_id,
                        config=config,
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