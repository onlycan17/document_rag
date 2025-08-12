import os
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

# 모듈 전역 캐시/가드
_BOOTSTRAP_DONE: bool = False
_RESOLVED_MIDM: Path | None = None
_RESOLVED_GEMMA3N: Path | None = None
_RESOLVED_GEMMA2B: Path | None = None
_RESOLVED_EXAONE: Path | None = None
_RESOLVED_AX_VL: Path | None = None


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _local_models_root() -> Path:
    # 절대 경로 /local_models 지원 + 환경변수 LOCAL_MODELS_DIR 우선
    env_path = os.getenv("LOCAL_MODELS_DIR")
    if env_path:
        return Path(env_path)
    # 프로젝트 로컬 디렉토리 우선
    prj_local = _project_root() / "local_models"
    if prj_local.exists():
        return prj_local
    return Path("/local_models")


def _midm_path() -> Path:
    # LOCAL_LLM_GGUF_PATH가 있으면 그 경로를 우선 보장
    env_path = os.getenv("LOCAL_LLM_GGUF_PATH")
    if env_path:
        return Path(env_path)
    # /local_models 우선 탐색
    lm = _local_models_root() / "korean" / "Midm-2.0-Base-Instruct-Q4_K_S.gguf"
    if lm.exists():
        return lm
    # 패턴 매칭으로 폭넓게 탐색 (예: 파일명이 약간 다른 경우)
    try:
        for p in _local_models_root().rglob("*.gguf"):
            name = p.name.lower()
            if "midm" in name and "instruct" in name:
                logger.info(f"🔎 Midm GGUF 발견: {p}")
                return p
    except Exception:
        pass
    return _project_root() / "models" / "korean" / "Midm-2.0-Base-Instruct-Q4_K_S.gguf"


def _gemma3n_dir_or_file() -> Path:
    """Gemma 3n 경로 탐색 (Transformers 디렉토리 우선, GGUF는 최후).

    우선순위:
    1) 환경변수 GEMMA_MULTIMODAL_DIR (config.json 존재해야 함)
    2) local_models 하위에서 config.json이 있는 디렉토리(예: gemma-3n-E4B-it-MLX-4bit)
    3) 표준 디렉토리 local_models/multimodal/gemma-3n-e4b
    4) 그 외 GGUF 파일 (텍스트 전용) – 멀티모달에는 부적합하지만 경로 제공
    """
    root = _local_models_root()

    # 0) 환경 변수 지정 우선
    env_dir = os.getenv("GEMMA_MULTIMODAL_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists() and p.is_dir() and (p / "config.json").exists():
            return p

    # 1) any directory with config.json and gemma keyword (디렉토리명에 MLX가 포함되어도 config.json이 있으면 허용)
    try:
        for cfg in root.rglob("config.json"):
            d = cfg.parent
            name = d.name.lower()
            if "gemma" in name and ("3n" in name or "e4b" in name):
                logger.info(f"🔎 Gemma 3n Transformers 디렉토리 후보: {d}")
                return d
    except Exception:
        pass

    # 2) 표준 디렉토리 경로
    lm = root / "multimodal" / "gemma-3n-e4b"
    if lm.exists():
        return lm
    try:
        for d in root.rglob("*"):
            if d.is_dir():
                lname = d.name.lower()
                if "gemma-3n-e4b" in lname:
                    logger.info(f"🔎 Gemma 3n 디렉토리 발견: {d}")
                    return d
    except Exception:
        pass

    # 3) GGUF 파일(최후)
    try:
        for p in root.rglob("*.gguf"):
            name = p.name.lower()
            if "gemma" in name and ("3n" in name or "e4b" in name):
                logger.info(f"🔎 Gemma 3n GGUF 발견: {p}")
                return p
    except Exception:
        pass

    return _project_root() / "models" / "multimodal" / "gemma-3n-e4b"


def _gemma2b_dir() -> Path:
    lm = _local_models_root() / "multimodal" / "gemma-2-2b-it"
    if lm.exists():
        return lm
    try:
        for d in _local_models_root().rglob("*"):
            if d.is_dir() and "gemma-2-2b-it" in d.name.lower():
                logger.info(f"🔎 Gemma 2-2b-it 디렉토리 발견: {d}")
                return d
    except Exception:
        pass
    return _project_root() / "models" / "multimodal" / "gemma-2-2b-it"


def _exaone_dir() -> Path:
    # 환경변수 우선
    env_dir = os.getenv("EXAONE_LOCAL_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    lm = _local_models_root() / "local" / "exaone-4.0-32b"
    if lm.exists():
        return lm
    try:
        for d in _local_models_root().rglob("*"):
            if d.is_dir() and "exaone" in d.name.lower() and "32b" in d.name.lower():
                logger.info(f"🔎 EXAONE 디렉토리 발견: {d}")
                return d
    except Exception:
        pass
    return _project_root() / "models" / "local" / "exaone-4.0-32b"


def _ax_vl_dir() -> Path:
    root = _local_models_root()
    # 환경변수 우선
    env_dir = os.getenv("AX_VL_LOCAL_DIR") or os.getenv("AX_MULTIMODAL_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    # 표준 위치
    lm = root / "multimodal" / "A.X-4.0-VL-Light"
    if lm.exists():
        return lm
    try:
        for d in root.rglob("*"):
            if d.is_dir() and ("a.x-4.0-vl-light" in d.name.lower() or "ax-4.0-vl-light" in d.name.lower()):
                logger.info(f"🔎 A.X 4.0 VL Light 디렉토리 발견: {d}")
                return d
    except Exception:
        pass
    return _project_root() / "models" / "multimodal" / "A.X-4.0-VL-Light"


def get_midm_path() -> Path:
    global _RESOLVED_MIDM
    if _RESOLVED_MIDM and _RESOLVED_MIDM.exists():
        return _RESOLVED_MIDM
    _RESOLVED_MIDM = _midm_path()
    return _RESOLVED_MIDM


def get_gemma_dir(prefer_3n: bool = True) -> Path:
    global _RESOLVED_GEMMA3N, _RESOLVED_GEMMA2B
    if prefer_3n:
        if _RESOLVED_GEMMA3N and _RESOLVED_GEMMA3N.exists():
            return _RESOLVED_GEMMA3N
        d = _gemma3n_dir_or_file()
        if d.exists():
            _RESOLVED_GEMMA3N = d
            return d
        if _RESOLVED_GEMMA2B and _RESOLVED_GEMMA2B.exists():
            return _RESOLVED_GEMMA2B
        d2 = _gemma2b_dir()
        _RESOLVED_GEMMA2B = d2
        return d2
    else:
        if _RESOLVED_GEMMA2B and _RESOLVED_GEMMA2B.exists():
            return _RESOLVED_GEMMA2B
        d2 = _gemma2b_dir()
        _RESOLVED_GEMMA2B = d2
        return d2


def get_exaone_dir() -> Path:
    global _RESOLVED_EXAONE
    if _RESOLVED_EXAONE and _RESOLVED_EXAONE.exists():
        return _RESOLVED_EXAONE
    _RESOLVED_EXAONE = _exaone_dir()
    return _RESOLVED_EXAONE


def get_ax_vl_dir() -> Path:
    global _RESOLVED_AX_VL
    if _RESOLVED_AX_VL and _RESOLVED_AX_VL.exists():
        return _RESOLVED_AX_VL
    _RESOLVED_AX_VL = _ax_vl_dir()
    return _RESOLVED_AX_VL


def ensure_models_available(download_exaone: bool = False) -> None:
    """필수(및 선택적) 모델이 없으면 다운로드합니다.

    - Midm-2.0 GGUF (필수, 지능형 이미지 추출의 한국어 주제 추출)
    - Gemma 3n e4b (권장) / 접근 불가 시 2b-it 폴백 다운로드 시도
    - EXAONE 4.0 32B (선택, 거대 모델) -> download_exaone=True일 때만
    """
    try:
        from scripts import download_models as dm
    except Exception as e:  # pragma: no cover
        logger.warning(f"모델 다운로드 스크립트를 불러오지 못했습니다: {e}")
        return

    # 0) 재실행 가드
    global _BOOTSTRAP_DONE
    if _BOOTSTRAP_DONE:
        logger.debug("모델 부트스트랩은 이미 완료됨")
        return

    # 1) Midm-2.0 (로컬 재탐색 포함)
    midm = get_midm_path()
    if not midm.exists():
        logger.info("🇰🇷 Midm-2.0 모델이 없어 자동 다운로드를 시작합니다...")
        try:
            ok = dm.download_midm_korean_model()
            if not ok:
                logger.warning("Midm-2.0 다운로드 실패")
        except Exception as e:
            logger.warning(f"Midm-2.0 다운로드 중 오류: {e}")
    else:
        logger.info(f"✅ Midm-2.0 로컬 모델 사용: {midm}")

    # 2) Gemma 멀티모달 (우선 3n-e4b, 실패 시 2b-it)
    if not get_gemma_dir(prefer_3n=True).exists():
        logger.info("🖼️ Gemma 멀티모달 모델이 없어 자동 다운로드를 시도합니다 (3n-e4b → 2b-it 폴백)...")
        # 우선 스크립트에 정의된 항목 호출 (repo는 3n-e4b로 설정되어 있어야 함)
        ok_3n = False
        try:
            ok_3n = dm.download_gemma_multimodal_model()
        except Exception as e:
            logger.warning(f"Gemma 3n-e4b 다운로드 중 오류: {e}")

        if not ok_3n:
            # 폴백: 2-2b-it을 직접 시도
            try:
                from huggingface_hub import snapshot_download
                target = _gemma2b_dir()
                snapshot_download(
                    repo_id="google/gemma-2-2b-it",
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                logger.info("✅ Gemma 2-2b-it 폴백 다운로드 완료")
            except Exception as e:
                logger.warning(f"Gemma 2-2b-it 폴백 다운로드 실패: {e}")
    else:
        found = get_gemma_dir(prefer_3n=True)
        logger.info(f"✅ Gemma 로컬 모델 사용: {found}")

    # 3) EXAONE (옵션)
    if download_exaone and not get_exaone_dir().exists():
        logger.info("🧠 EXAONE 4.0 32B 모델이 없어 자동 다운로드를 시작합니다...")
        try:
            ok = dm.download_exaone_model()
            if not ok:
                logger.warning("EXAONE 다운로드 실패")
        except Exception as e:
            logger.warning(f"EXAONE 다운로드 중 오류: {e}")

    _BOOTSTRAP_DONE = True


def preload_models() -> None:
    """서버 기동 시 모델을 미리 로드하여 초기 지연을 줄입니다.

    너무 무겁지 않게 각 모델을 가볍게 인스턴스화만 수행합니다.
    실패해도 치명적이지 않으며, 런타임 시 다시 로드됩니다.
    """
    # Midm-2.0
    try:
        midm = get_midm_path()
        if midm.exists():
            from src.utils.korean_text_model import KoreanTextModel
            _ = KoreanTextModel(model_path=str(midm), n_ctx=1024, n_threads=int(os.getenv("LOCAL_LLM_THREADS", "2")))
            logger.info("✅ Midm-2.0 사전 로드 완료")
    except Exception as e:
        logger.warning(f"Midm-2.0 사전 로드 실패(무시): {e}")

    # Gemma 멀티모달 (3n-e4b 또는 2-2b-it 어느 쪽이든)
    try:
        gdir = get_gemma_dir(prefer_3n=True)
        if gdir.exists():
            from src.utils.gemma_multimodal import GemmaMultimodalModel
            _ = GemmaMultimodalModel(model_path=str(gdir), load_in_4bit=False, max_memory_gb=4)
            logger.info("✅ Gemma 멀티모달 사전 로드 완료")
    except Exception as e:
        logger.warning(f"Gemma 사전 로드 실패(무시): {e}")


