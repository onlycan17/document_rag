"""
SKT A.X 4.0 VL Light 멀티모달 모델 - 이미지 분석 및 주제 관련성 판단

Hugging Face: skt/A.X-4.0-VL-Light

참고: https://huggingface.co/skt/A.X-4.0-VL-Light
"""

import os
import logging
import warnings
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import json

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)
from config import settings


logger = logging.getLogger(__name__)


class AXMultimodalModel:
    """SKT A.X 4.0 VL Light를 사용한 멀티모달 이미지 분석"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_memory_gb: int = 8,
    ):
        # 전역 torch.classes 경고 억제 설정 (A.X 모델 전용)
        self._setup_torch_warnings_suppression()
        
        self.model_path = self._get_model_path(model_path)
        self.device = self._setup_device(device)
        self.max_memory_gb = max_memory_gb

        self.model = None
        self.processor = None
        self._load_model()

    def _setup_torch_warnings_suppression(self):
        """torch.classes 및 PyTorch 관련 경고 메시지 전역 억제 설정 (강화된 버전)"""
        try:
            # 환경 변수를 통한 PyTorch 로깅 완전 억제
            os.environ.setdefault('TORCH_LOG_LEVEL', 'ERROR')
            os.environ.setdefault('PYTORCH_JIT_LOG_LEVEL', 'ERROR')
            os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
            os.environ.setdefault('PYTORCH_KERNEL_WARN', '0')
            os.environ.setdefault('TORCH_SHOW_CPP_STACKTRACES', '0')
            
            # Python warnings 모듈을 통한 포괄적 억제
            warnings.filterwarnings("ignore", message=".*torch.classes.*")
            warnings.filterwarnings("ignore", message=".*torch.ops.*") 
            warnings.filterwarnings("ignore", message=".*torch.jit.*")
            warnings.filterwarnings("ignore", message=".*__path__._path.*")
            warnings.filterwarnings("ignore", message=".*Examining the path.*")
            warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*classes.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*ops.*")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*torch.*")
            
            # PyTorch 내부 경고 시스템 강화 억제
            import torch._C
            if hasattr(torch._C, '_jit_set_emit_warnings'):
                try:
                    torch._C._jit_set_emit_warnings(False)
                except Exception:
                    pass
            
            # 추가 PyTorch 내부 경고 제어
            if hasattr(torch._C, '_set_print_stacktraces_on_fatal_signal'):
                try:
                    torch._C._set_print_stacktraces_on_fatal_signal(False)
                except Exception:
                    pass
                    
            if hasattr(torch._C, '_set_print_warn'):
                try:
                    torch._C._set_print_warn(False)
                except Exception:
                    pass
            
            # torch.classes 관련 로깅 레벨 강화 조정
            import logging
            torch_loggers = [
                'torch', 'torch.jit', 'torch.fx', 'torch._C', 'torch.nn',
                'transformers.modeling_utils', 'transformers.tokenization_utils',
                'transformers.models', 'transformers.configuration_utils'
            ]
            
            for logger_name in torch_loggers:
                try:
                    logging.getLogger(logger_name).setLevel(logging.ERROR)
                except Exception:
                    pass
            
            # transformers 모듈의 구체적 억제
            try:
                import transformers.utils.logging
                transformers.utils.logging.set_verbosity_error()
                transformers.utils.logging.disable_default_handler()
                transformers.utils.logging.disable_propagation()
            except Exception:
                pass
            
            logger.debug("🔇 A.X 모델 torch.classes 경고 강화 억제 설정 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ torch.classes 경고 강화 억제 설정 중 오류: {e}")

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
            # torch.classes 경고 메시지 억제 설정
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch.classes.*")
                warnings.filterwarnings("ignore", message=".*torch.ops.*")
                warnings.filterwarnings("ignore", message=".*torch.jit.*")
                warnings.filterwarnings("ignore", category=UserWarning, message=".*classes.*")
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
                
                model_id = str(self.model_path)
                p = Path(model_id)
                
                logger.info(f"🔇 PyTorch 클래스 관련 경고 메시지 억제 활성화")
                
                # torch.classes 관련 특별한 억제 조치
                import torch._C
                if hasattr(torch._C, '_jit_set_emit_warnings'):
                    try:
                        torch._C._jit_set_emit_warnings(False)
                        logger.debug("🔇 torch JIT 경고 메시지 억제")
                    except Exception:
                        pass
                
                # PyTorch의 클래스 등록 관련 로깅 레벨 조정
                torch_logger = logging.getLogger('torch')
                original_torch_level = torch_logger.level
                torch_logger.setLevel(logging.ERROR)
                
                # transformers의 클래스 관련 로깅도 억제
                transformers_logger = logging.getLogger('transformers')
                original_transformers_level = transformers_logger.level
                transformers_logger.setLevel(logging.ERROR)
            
            # 로컬 모델의 경우 transformers_modules 문제 해결을 위한 특별 처리
            if p.exists():
                logger.info(f"🔄 로컬 모델 로드 (transformers_modules 우회): {model_id}")
                
                # transformers_modules 문제 해결: 직접 모듈 로딩 방식
                import sys
                import importlib.util
                from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
                
                # 1단계: 로컬 모듈을 sys.path에 추가
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                
                try:
                    # 2단계: transformers_modules 문제 완전 해결 - 패키지 구조 생성
                    logger.info("   커스텀 모듈 직접 로딩 중 (상대 임포트 해결)...")
                    
                    # 환경변수 설정으로 HuggingFace 오프라인 모드 강제
                    import os
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    
                    # HuggingFace 캐시 완전 우회
                    os.environ["HF_HOME"] = str(p.parent / "temp_hf_cache")
                    os.environ["TRANSFORMERS_CACHE"] = str(p.parent / "temp_hf_cache")
                    
                    # transformers_modules 디렉토리 강제 무시
                    import sys
                    if hasattr(sys, 'modules'):
                        # transformers_modules 관련 모든 모듈 제거
                        modules_to_remove = [k for k in sys.modules.keys() if 'transformers_modules' in k]
                        for mod in modules_to_remove:
                            del sys.modules[mod]
                            logger.info(f"   🗑️ 제거된 모듈: {mod}")
                    
                    # HuggingFace 캐시 완전 삭제 및 재생성
                    import shutil
                    hf_cache_dirs = [
                        Path.home() / ".cache" / "huggingface" / "transformers",
                        Path.home() / ".cache" / "huggingface" / "hub",
                        p.parent / "temp_hf_cache"
                    ]
                    
                    for cache_dir in hf_cache_dirs:
                        if cache_dir.exists():
                            try:
                                shutil.rmtree(cache_dir)
                                logger.info(f"   🗑️ HF 캐시 삭제: {cache_dir}")
                            except Exception as e:
                                logger.warning(f"   ⚠️ 캐시 삭제 실패: {e}")
                    
                    # 임시 패키지 디렉토리 생성
                    temp_package_dir = p.parent / "temp_ax4vl_package"
                    temp_package_dir.mkdir(exist_ok=True)
                    
                    # __init__.py 파일 생성
                    init_file = temp_package_dir / "__init__.py"
                    if not init_file.exists():
                        init_file.write_text("# Auto-generated package for A.X-4.0-VL-Light\n")
                    
                    # 필요한 py 파일들을 임시 패키지로 복사 (호환성 수정 포함)
                    import shutil
                    for py_file in ["configuration_ax4vl.py", "modeling_ax4vl.py", "processing_ax4vl.py", "image_processing_ax4vl.py"]:
                        src = p / py_file
                        dst = temp_package_dir / py_file
                        if src.exists() and not dst.exists():
                            if py_file == "processing_ax4vl.py":
                                # processing_ax4vl.py의 호환성 문제 수정
                                with open(src, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                # transformers 4.55.0 호환성 강화 패치
                                content = content.replace(
                                    "from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, _validate_images_text_input_order",
                                    "from transformers.processing_utils import ProcessingKwargs, ProcessorMixin\n"
                                    "# transformers 4.55.0 호환성: _validate_images_text_input_order 함수 구현\n"
                                    "def _validate_images_text_input_order(images, text):\n"
                                    "    \"\"\"transformers 4.55.0 호환성을 위한 검증 함수\"\"\"\n"
                                    "    # 기본적인 검증만 수행\n"
                                    "    if images is None and text is None:\n"
                                    "        raise ValueError('Either images or text must be provided')\n"
                                    "    return images, text"
                                )
                                
                                # 추가 호환성 패치: 새로운 transformers API 대응
                                content = content.replace(
                                    "from transformers.feature_extraction_utils import BatchFeature",
                                    "try:\n"
                                    "    from transformers.feature_extraction_utils import BatchFeature\n"
                                    "except ImportError:\n"
                                    "    from transformers.utils import BatchFeature"
                                )
                                
                                with open(dst, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                logger.info(f"     📝 {py_file} 호환성 패치 적용")
                            else:
                                shutil.copy2(src, dst)
                    
                    # sys.path에 임시 패키지의 부모 디렉토리 추가
                    package_parent = str(temp_package_dir.parent)
                    if package_parent not in sys.path:
                        sys.path.insert(0, package_parent)
                    
                    # 패키지로 모듈 임포트 (상대 임포트 해결됨)
                    logger.info("   패키지 기반 모듈 임포트 중...")
                    
                    # configuration 모듈
                    import temp_ax4vl_package.configuration_ax4vl as config_module
                    sys.modules["configuration_ax4vl"] = config_module
                    
                    # modeling 모듈
                    import temp_ax4vl_package.modeling_ax4vl as model_module  
                    sys.modules["modeling_ax4vl"] = model_module
                    
                    # processing 모듈
                    import temp_ax4vl_package.processing_ax4vl as proc_module
                    sys.modules["processing_ax4vl"] = proc_module
                    
                    # image_processing 모듈 (필요한 경우)
                    try:
                        import temp_ax4vl_package.image_processing_ax4vl as img_proc_module
                        sys.modules["image_processing_ax4vl"] = img_proc_module
                    except Exception:
                        pass
                    
                    logger.info("   ✅ 커스텀 모듈 로딩 완료 (패키지 방식)")
                    
                    # 3단계: AutoConfig로 모듈 강제 등록 (기존 등록 덮어쓰기)
                    try:
                        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
                        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
                        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
                        
                        # 기존 등록 제거
                        if "a.x-4-vl" in CONFIG_MAPPING._extra_content:
                            del CONFIG_MAPPING._extra_content["a.x-4-vl"]
                        
                        # 새로운 등록 (강제)
                        CONFIG_MAPPING.register("a.x-4-vl", config_module.AX4VLConfig, exist_ok=True)
                        MODEL_FOR_CAUSAL_LM_MAPPING.register(config_module.AX4VLConfig, model_module.AX4VLForConditionalGeneration, exist_ok=True)
                        PROCESSOR_MAPPING.register(config_module.AX4VLConfig, proc_module.AX4VLProcessor, exist_ok=True)
                        
                        # 추가: 직접 매핑에도 등록
                        AutoConfig.register("a.x-4-vl", config_module.AX4VLConfig)
                        AutoModelForCausalLM.register(config_module.AX4VLConfig, model_module.AX4VLForConditionalGeneration)
                        AutoProcessor.register(config_module.AX4VLConfig, proc_module.AX4VLProcessor)
                        
                        logger.info("   ✅ AutoModel 클래스 강제 등록 완료")
                    except Exception as reg_error:
                        logger.warning(f"   클래스 등록 건너뜀: {reg_error}")
                    
                    # 4단계: 로컬 파일에서 프로세서 로드 (transformers_modules 완전 우회)
                    self.processor = AutoProcessor.from_pretrained(
                        model_id,
                        trust_remote_code=False,  # 로컬 등록된 클래스만 사용
                        local_files_only=True,
                        use_fast=False,
                        force_download=False,
                        resume_download=False,
                        cache_dir=None,  # 캐시 완전 무시
                        revision=None
                    )
                    logger.info("   ✅ 프로세서 로드 성공")
                    logger.info(f"   프로세서 타입: {type(self.processor).__name__}")
                    logger.info(f"   프로세서 메서드: {[m for m in dir(self.processor) if not m.startswith('_')][:10]}")
                    
                except Exception as direct_error:
                    logger.warning(f"   직접 로딩 실패: {direct_error}")
                    logger.info("   HuggingFace 원격에서 재시도...")
                    
                    # 폴백: HuggingFace에서 다운로드 (비활성화)
                    logger.error("   ❌ 로컬 모델 로딩 실패 - 원격 다운로드 비활성화됨")
                    raise Exception("로컬 모델 로딩 필수 - 원격 다운로드 비허용")
            else:
                logger.info(f"🔄 HuggingFace에서 모델 로드: {model_id}")
                # HuggingFace ID로 직접 로드 (비활성화)
                logger.error(f"❌ 원격 모델 로드 비허용: {model_id}")
                raise Exception("로컬 모델만 허용됨")

            # dtype/장치 설정
            if self.device == "cuda":
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif self.device == "mps":
                # MPS는 float16 권장 (메모리 절감 및 속도 향상)
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            load_kwargs = {
                "torch_dtype": torch_dtype,
                "use_safetensors": True,
            }
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {0: f"{self.max_memory_gb}GB", "cpu": f"{self.max_memory_gb * 2}GB"}

            # 모델 로딩도 같은 방식으로 처리
            if p.exists():
                try:
                    # 로컬에서 모델 로드 (transformers_modules 완전 우회)
                    logger.info("   모델 가중치 로딩 중...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=False,  # 로컬 등록된 클래스만 사용
                        local_files_only=True,
                        force_download=False,
                        resume_download=False,
                        cache_dir=None,  # 캐시 완전 무시
                        revision=None,
                        **load_kwargs,
                    )
                    logger.info("   ✅ 모델 가중치 로드 성공")
                    
                except Exception as local_error:
                    logger.error(f"   ❌ 로컬 모델 로드 실패: {local_error}")
                    logger.error("   원격 다운로드 비활성화됨")
                    raise Exception("로컬 모델 로딩 필수")
            else:
                logger.error(f"❌ 원격 모델 ID 비허용: {model_id}")
                raise Exception("로컬 모델만 허용됨")

            # meta tensor 문제 해결: to_empty() 사용
            if self.device != "cuda":
                try:
                    # PyTorch 2.0+ meta tensor 호환성 처리
                    if hasattr(self.model, '_modules'):
                        # 모든 파라미터가 meta device에 있는지 확인
                        has_meta_params = any(p.device.type == 'meta' for p in self.model.parameters())
                        if has_meta_params:
                            logger.info(f"   🔄 Meta tensor 감지 - to_empty() 방식으로 디바이스 이동: {self.device}")
                            # to_empty()를 사용하여 meta tensor 문제 해결
                            self.model = self.model.to_empty(device=self.device)
                        else:
                            # 일반적인 디바이스 이동
                            self.model = self.model.to(self.device)
                    else:
                        self.model = self.model.to(self.device)
                    logger.info(f"   ✅ 모델 디바이스 이동 완료: {self.device}")
                except Exception as device_error:
                    logger.warning(f"   ⚠️ 디바이스 이동 실패 ({self.device}): {device_error}")
                    # MPS 실패 시 CPU로 폴백
                    if self.device == "mps":
                        logger.info("   🔄 MPS 실패 - CPU로 폴백")
                        self.device = "cpu"
                        try:
                            self.model = self.model.to("cpu")
                            logger.info("   ✅ CPU 디바이스 이동 성공")
                        except Exception as cpu_error:
                            logger.error(f"   ❌ CPU 이동도 실패: {cpu_error}")
                            raise cpu_error
                    else:
                        raise device_error

            self.model.eval()
            logger.info("✅ A.X 멀티모달 모델 로드 완료!")
            
            # 로깅 레벨 복원
            try:
                if 'torch_logger' in locals():
                    torch_logger.setLevel(original_torch_level)
                if 'transformers_logger' in locals():
                    transformers_logger.setLevel(original_transformers_level)
                logger.debug("🔄 로깅 레벨 복원 완료")
            except Exception:
                pass

        except Exception as e:
            logger.error(f"❌ A.X 모델 로드 실패: {e}")
            logger.error("💡 원격 다운로드가 비활성화되어 로컬 모델만 사용 가능합니다")
            self.model = None
            self.processor = None

    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """입력 텐서를 디바이스로 안전하게 이동한다.
        - input_ids/labels: Long(Int64)
        - attention_mask/position_ids: 정수 유지
        - 그 외 부동소수 텐서는 모델 dtype으로 캐스팅
        """
        moved: Dict[str, Any] = {}
        target_dtype = getattr(self.model, 'dtype', None)
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
                continue
            # 정수형 보존 및 강제 형변환 규칙
            if key in ("input_ids", "labels"):
                moved[key] = value.to(self.device, dtype=torch.long)
                continue
            if key in ("attention_mask", "position_ids", "token_type_ids"):
                # 정수 마스크 등은 dtype 유지
                moved[key] = value.to(self.device)
                continue
            # 부동소수 텐서는 모델 dtype으로
            if value.is_floating_point() and target_dtype is not None:
                moved[key] = value.to(self.device, dtype=target_dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

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
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            # 환경변수로 이미지 최대 해상도 조정 가능 (기본 896)
            try:
                max_img = int(os.getenv("AX_IMAGE_MAX_SIZE", "896"))
            except Exception:
                max_img = 896
            image = self._resize_image(image, max_size=max_img)

            prompt_parts = ["이미지를 분석하여 한국어로 답변하세요.\n"]
            if document_topic:
                prompt_parts.append(f"문서 주제: {document_topic.get('main_topic', 'Unknown')}\n")
                if document_topic.get("keywords"):
                    prompt_parts.append(f"키워드: {', '.join(document_topic['keywords'][:5])}\n")
            if context:
                prompt_parts.append(f"컨텍스트: {context[:200]}\n")
            prompt_parts.append(
                """
출력은 반드시 아래 JSON 한 줄로만:
{"type": "<one-of:text|diagram|photo|chart|table|map|document>", "content": "<핵심 요약>", "relevance": <0~1 숫자>, "text_only": <true|false>}
주의:
- JSON 외의 설명/서문/코드블록 금지
- type은 반드시 목록 중 하나만 사용 (동의어 금지)
- relevance는 0~1 사이 소수점으로 표기
"""
            )
            prompt = "".join(prompt_parts)

            # Chat template 기반으로 이미지 토큰이 포함된 입력 생성
            inputs = None
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                # apply_chat_template가 있는 경우 반드시 사용하여 이미지 토큰 삽입
                if hasattr(self.processor, "apply_chat_template"):
                    templated_text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    templated_text = prompt  # 폴백 (이미지 토큰 미삽입 가능성 있음)

                inputs = self.processor(
                    text=[templated_text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
                # 일부 백엔드에서 token_type_ids 미지원 → 안전하게 제거
                if isinstance(inputs, dict):
                    inputs.pop("token_type_ids", None)
                logger.debug("chat_template 기반 입력 생성 완료")
            except Exception as e_inputs:
                logger.error(f"입력 생성 실패: {e_inputs}")
                raise

            if self.device != "cpu":
                inputs = self._move_inputs_to_device(inputs)

            # torch.classes 경고 억제를 위한 추론 시 설정
            with warnings.catch_warnings(), torch.inference_mode():
                warnings.filterwarnings("ignore", message=".*torch.classes.*")
                warnings.filterwarnings("ignore", message=".*torch.ops.*")
                warnings.filterwarnings("ignore", category=UserWarning, message=".*classes.*")
                try:
                    # 환경변수로 생성 토큰 상한 조정 (기본 256→환경값 우선)
                    max_new = int(os.getenv("AX_MAX_NEW_TOKENS", "256"))
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        temperature=settings.temperature,
                        do_sample=True,
                        top_p=0.9,
                    )
                except Exception as gen_e:
                    logger.error(f"모델 생성 중 오류 발생: {gen_e}")
                    return {"error": f"Model generation failed: {gen_e}"}

            # outputs 검증 및 안전한 디코딩
            if outputs is None:
                logger.error("모델 생성 실패: outputs가 None입니다")
                return {"error": "Model generation failed: outputs is None"}
            
            if not hasattr(outputs, '__len__') or len(outputs) == 0:
                logger.error("모델 생성 실패: outputs가 비어있습니다") 
                return {"error": "Model generation failed: empty outputs"}
                
            # outputs의 타입과 내용을 확인
            if not torch.is_tensor(outputs) and not isinstance(outputs, (list, tuple)):
                logger.error(f"모델 생성 실패: 예상치 못한 outputs 타입 {type(outputs)}")
                return {"error": f"Unexpected outputs type: {type(outputs)}"}

            try:
                # 1) Processor의 batch_decode 우선
                if hasattr(self.processor, 'batch_decode'):
                    decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
                    if decoded_outputs and len(decoded_outputs) > 0:
                        response = decoded_outputs[0]
                    else:
                        logger.error("batch_decode 결과가 비어있습니다")
                        return {"error": "Empty batch_decode result"}
                # 2) Processor 내부 tokenizer가 있으면 그 batch_decode 사용
                elif hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'batch_decode'):
                    decoded_outputs = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    if decoded_outputs and len(decoded_outputs) > 0:
                        response = decoded_outputs[0]
                    else:
                        logger.error("tokenizer.batch_decode 결과가 비어있습니다")
                        return {"error": "Empty tokenizer batch_decode result"}
                # 3) 마지막 폴백: decode 단건 호출
                else:
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
            except IndexError as e:
                logger.error(f"디코딩 인덱스 오류: {e}, outputs 길이: {len(outputs) if hasattr(outputs, '__len__') else 'unknown'}")
                return {"error": f"Decoding index error: {e}"}
            except Exception as e:
                logger.error(f"디코딩 오류: {e}")
                return {"error": f"Decoding error: {e}"}

            if prompt in response:
                response = response.replace(prompt, "").strip()

            parsed = self._parse_response(response, document_topic)

            # 메모리 정리
            try:
                del inputs, outputs, image
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps" and hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
            except Exception:
                pass

            return parsed

        except Exception as e:
            import traceback
            logger.error(f"이미지 분석 실패(A.X): {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
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

        # JSON 우선 파싱 시도 (모델 프롬프트를 JSON으로 유도함)
        try:
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end].strip()
                data = json.loads(json_str)
                if isinstance(data, dict):
                    allowed_types = {
                        "text", "diagram", "photo", "chart", "table", "map", "document"
                    }
                    synonyms = {
                        "figure": "diagram",
                        "graph": "chart",
                        "plot": "chart",
                        "picture": "photo",
                        "image": "photo",
                        "scan": "document",
                        "page": "document",
                    }
                    t_val = str(data.get("type", "")).strip().lower()
                    t_val = synonyms.get(t_val, t_val)
                    if t_val in allowed_types:
                        result["image_type"] = t_val
                    c_val = data.get("content", "")
                    if isinstance(c_val, str) and c_val.strip():
                        result["content_description"] = c_val.strip()
                    r_val = data.get("relevance", None)
                    if isinstance(r_val, (int, float)):
                        score = float(r_val)
                        if score > 1.0:
                            score = score / 100.0
                        result["relevance_score"] = min(max(score, 0.0), 1.0)
                    to_val = data.get("text_only", None)
                    if isinstance(to_val, bool):
                        result["is_text_only"] = to_val
                    return result
        except Exception:
            # JSON 파싱 실패 시 라인 기반 파싱으로 폴백
            pass

        # 안전 파싱 헬퍼
        def _after_colon_or_tail(line: str, labels: List[str]) -> str:
            raw = line.strip()
            if ":" in raw:
                return raw.split(":", 1)[1].strip()
            # 라벨 접두어 제거 시도
            lowered = raw.lower()
            for label in labels:
                if lowered.startswith(label.lower()):
                    tail = raw[len(label):].strip(" -:\t")
                    return tail
            # 공백으로 두 토큰 이상이면 두 번째 토큰부터 tail
            parts = raw.split(None, 1)
            if len(parts) == 2:
                return parts[1].strip()
            return raw

        # 간단 파싱 (완전 방어형)
        for line in response.split("\n"):
            s = line.strip()
            if not s:
                continue
            lower_s = s.lower()
            try:
                if ("type:" in lower_s) or ("유형" in s):
                    t = _after_colon_or_tail(s, ["유형", "type"]).lower()
                    if "text" in t:
                        result["image_type"] = "text"
                        result["is_text_only"] = True
                    elif ("diagram" in t) or ("도표" in t):
                        result["image_type"] = "diagram"
                    elif ("photo" in t) or ("사진" in t):
                        result["image_type"] = "photo"
                    elif ("chart" in t) or ("그래프" in t):
                        result["image_type"] = "chart"
                elif ("relevance:" in lower_s) or ("관련도" in s):
                    import re
                    nums = re.findall(r"[\d.]+", s)
                    if nums:
                        score = float(nums[0])
                        if score > 1.0:
                            score = score / 100.0
                        result["relevance_score"] = min(max(score, 0.0), 1.0)
                elif ("content:" in lower_s) or ("내용" in s):
                    desc = _after_colon_or_tail(s, ["내용", "content"])
                    if desc:
                        result["content_description"] = desc
            except Exception:
                # 파싱 실패 시 해당 라인 스킵 (전역 실패 방지)
                continue

        if not result["content_description"] and response:
            for line in response.split("\n"):
                if len(line.strip()) > 10:
                    result["content_description"] = line.strip()
                    break

        # 휴리스틱 보정: 여전히 unknown이면 내용 기반으로 추정
        if result["image_type"] == "unknown":
            heur = (result["content_description"] or response).lower()
            if any(k in heur for k in ["표", "table"]):
                result["image_type"] = "table"
            elif any(k in heur for k in ["지도", "map"]):
                result["image_type"] = "map"
            elif any(k in heur for k in ["그래프", "chart", "graph", "plot"]):
                result["image_type"] = "chart"
            elif any(k in heur for k in ["도표", "diagram", "flow"]):
                result["image_type"] = "diagram"
            elif any(k in heur for k in ["사진", "photo", "image", "picture"]):
                result["image_type"] = "photo"
            elif result["is_text_only"]:
                result["image_type"] = "text"

        return result

    def classify_image(self, image_path: str) -> Tuple[bool, float]:
        if not self.model or not self.processor:
            return False, 0.5
        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            image = self._resize_image(image, max_size=512)
            prompt = (
                "이 이미지는 주로 텍스트인가요, 아니면 의미있는 이미지인가요?\n"
                "답변: text-only 또는 meaningful-image\n"
                "신뢰도: 0-1"
            )
            # chat template 기반 입력 생성으로 이미지 토큰 포함 보장
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            if hasattr(self.processor, "apply_chat_template"):
                templated_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                templated_text = prompt
            inputs = self.processor(
                text=[templated_text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            if isinstance(inputs, dict):
                inputs.pop("token_type_ids", None)
            if self.device != "cpu":
                inputs = self._move_inputs_to_device(inputs)
            
            # torch.classes 경고 억제를 위한 추론 시 설정
            with warnings.catch_warnings(), torch.inference_mode():
                warnings.filterwarnings("ignore", message=".*torch.classes.*")
                warnings.filterwarnings("ignore", message=".*torch.ops.*")
                warnings.filterwarnings("ignore", category=UserWarning, message=".*classes.*")
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=settings.temperature,
                        do_sample=False,
                    )
                except Exception as gen_e:
                    logger.error(f"이미지 분류 모델 생성 중 오류: {gen_e}")
                    return False, 0.5
            
            # outputs 검증 및 안전한 디코딩
            if outputs is None:
                logger.error("이미지 분류 실패: outputs가 None입니다")
                return False, 0.5
            
            if not hasattr(outputs, '__len__') or len(outputs) == 0:
                logger.error("이미지 분류 실패: outputs가 비어있습니다")
                return False, 0.5

            try:
                if hasattr(self.processor, 'batch_decode'):
                    decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
                    if decoded_outputs and len(decoded_outputs) > 0:
                        response = decoded_outputs[0]
                    else:
                        logger.error("이미지 분류 batch_decode 결과가 비어있습니다")
                        return False, 0.5
                else:
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
            except IndexError as e:
                logger.error(f"이미지 분류 디코딩 인덱스 오류: {e}")
                return False, 0.5
            except Exception as e:
                logger.error(f"이미지 분류 디코딩 오류: {e}")
                return False, 0.5
            text = response.lower()
            is_text_only = ("text-only" in text) or ("text only" in text)
            confidence = 0.8
            import re
            nums = re.findall(r"[\d.]+", text)
            if nums:
                confidence = float(nums[0])
                if confidence > 1:
                    confidence = confidence / 100
            # 메모리 정리
            try:
                del inputs, outputs, image
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps" and hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
            except Exception:
                pass

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


