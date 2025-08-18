import os
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

# 모듈 전역 캐시/가드
_BOOTSTRAP_DONE: bool = False
_RESOLVED_MIDM: Path | None = None
_RESOLVED_GGUF_ANY: Path | None = None
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


def _any_gguf_path(preferred_keywords: list[str] | None = None) -> Path:
    """local_models 이하에서 임의의 GGUF 파일을 탐색하여 반환.

    우선순위:
    1) 환경변수 LOCAL_LLM_GGUF_PATH
    2) local_models/**.gguf (preferred_keywords 스코어링 후 최상위)
    3) 프로젝트 기본 models/**.gguf (호환성)
    찾지 못하면 예상 경로 객체를 반환(존재하지 않을 수 있음).
    """
    # 1) 환경변수 우선
    env_path = os.getenv("LOCAL_LLM_GGUF_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2) local_models 재귀 탐색
    root = _local_models_root()
    candidates: list[Path] = []
    try:
        for p in root.rglob("*.gguf"):
            if p.is_file():
                candidates.append(p)
    except Exception:
        pass

    def _score(path: Path) -> int:
        name = path.name.lower()
        score = 0
        # 한국어/지시형 우선 키워드 가점
        for kw in ["korean", "ko", "instruct", "q4", "q5", "q8"]:
            if kw in name:
                score += 1
        if preferred_keywords:
            for kw in preferred_keywords:
                if kw.lower() in name:
                    score += 2
        return score

    if candidates:
        # Midm 제외 선호: 다른 후보가 있으면 midm 포함 항목 제거
        non_midm = [p for p in candidates if "midm" not in p.name.lower()]
        pool = non_midm if non_midm else candidates
        best = sorted(pool, key=_score, reverse=True)[0]
        logger.info(f"🔎 GGUF 모델 선택: {best}")
        return best

    # 3) 프로젝트 기본 디렉토리 Fallback 탐색
    proj = _project_root()
    try:
        for p in (proj / "models").rglob("*.gguf"):
            if p.is_file():
                logger.info(f"🔎 GGUF 후보(프로젝트): {p}")
                return p
    except Exception:
        pass

    # 아무것도 없을 경우 예상 경로 반환 (존재하지 않을 수 있음)
    return proj / "models" / "local" / "model.gguf"


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


def get_gguf_path(preferred_keywords: list[str] | None = None) -> Path:
    """임의 GGUF 모델 경로 반환(캐시 포함)."""
    global _RESOLVED_GGUF_ANY
    if _RESOLVED_GGUF_ANY and _RESOLVED_GGUF_ANY.exists():
        return _RESOLVED_GGUF_ANY
    _RESOLVED_GGUF_ANY = _any_gguf_path(preferred_keywords)
    return _RESOLVED_GGUF_ANY


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
    """로컬 모델 가용성 확인(다운로드 강제하지 않음).

    - GGUF: local_models 내 임의 모델 사용 가능
    - Gemma/EXAONE: 존재 시만 사용, 네트워크 환경에서는 다운로드 생략
    """
    # GGUF 존재 확인
    gguf = get_gguf_path()
    if gguf and gguf.exists():
        logger.info(f"✅ 로컬 GGUF 모델 감지: {gguf}")
    else:
        logger.warning("⚠️ 로컬 GGUF 모델을 찾지 못했습니다. LOCAL_LLM_GGUF_PATH를 설정하거나 local_models에 배치하세요.")

    # 선택 모델(Transformers) 경로 로그만
    exa = get_exaone_dir()
    if exa.exists():
        logger.info(f"🔎 EXAONE 로컬 디렉토리 확인: {exa}")
    ax = get_ax_vl_dir()
    if ax.exists():
        logger.info(f"🔎 A.X VL 로컬 디렉토리 확인: {ax}")

    # 0) 재실행 가드
    global _BOOTSTRAP_DONE
    if _BOOTSTRAP_DONE:
        logger.debug("모델 부트스트랩은 이미 완료됨")
        return

    # 다운로드는 수행하지 않음: 오프라인/로컬 환경 우선
    # 멀티모달/EXAONE은 경로가 있을 때만 사용하도록 로그만 출력
    found = get_gemma_dir(prefer_3n=True)
    if found.exists():
        label = "A.X 4.0 VL Light" if ("a.x-4.0-vl-light" in str(found).lower() or "ax-4.0-vl-light" in str(found).lower()) else "Gemma"
        logger.info(f"✅ {label} 멀티모달 모델 사용 가능: {found}")

    _BOOTSTRAP_DONE = True


def preload_models() -> None:
    """서버 기동 시 모델을 미리 로드하여 초기 지연을 줄입니다.

    너무 무겁지 않게 각 모델을 가볍게 인스턴스화만 수행합니다.
    실패해도 치명적이지 않으며, 런타임 시 다시 로드됩니다.
    
    파일 기반 잠금을 사용하여 다중 프로세스 환경에서 개별 모델 로딩 동기화
    """
    import fcntl
    import time
    from pathlib import Path
    
    # 개별 모델 로딩 동기화를 위한 잠금 파일
    lock_file_path = Path(__file__).parent.parent.parent / ".preload_models_lock"
    preload_done_file = Path(__file__).parent.parent.parent / ".preload_models_done"
    
    # 이미 사전 로드가 완료되었다면 스킵
    if preload_done_file.exists():
        logger.debug("모델 사전 로드가 이미 완료됨 (파일 확인)")
        return
    
    # 잠금 파일로 동기화
    try:
        with open(lock_file_path, 'w') as lock_file:
            try:
                # 비블로킹 잠금 시도
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # 잠금 획득 성공 - 이미 완료되었는지 다시 확인
                if preload_done_file.exists():
                    logger.debug("모델 사전 로드가 다른 프로세스에서 완료됨")
                    return
                
                logger.info("🔄 프로세스 동기화 모델 사전 로드 시작...")
                
                # 로컬 GGUF 사전 로드
                try:
                    from src.utils.model_bootstrap import get_gguf_path
                    gguf = get_gguf_path()
                    if gguf.exists():
                        from src.utils.korean_text_model import KoreanTextModel
                        _ = KoreanTextModel(model_path=str(gguf), n_ctx=1024, n_threads=int(os.getenv("LOCAL_LLM_THREADS", "2")))
                        logger.info("✅ GGUF 사전 로드 완료")
                except Exception as e:
                    logger.warning(f"GGUF 사전 로드 실패(무시): {e}")

                # A.X 멀티모달 모델
                try:
                    ax_dir = get_ax_vl_dir()
                    if ax_dir.exists():
                        from src.utils.ax_multimodal import AXMultimodalModel
                        _ = AXMultimodalModel(model_path=str(ax_dir), device="auto", max_memory_gb=4)
                        logger.info("✅ A.X 멀티모달 사전 로드 완료")
                except Exception as e:
                    logger.warning(f"A.X 사전 로드 실패(무시): {e}")
                
                # 완료 표시 파일 생성
                preload_done_file.touch()
                logger.info("🎉 프로세스 동기화 모델 사전 로드 완료")
                
            except BlockingIOError:
                # 다른 프로세스가 이미 잠금 보유 중 - 완료될 때까지 대기
                logger.debug("다른 프로세스가 모델 사전 로드 중... 대기")
                max_wait_time = 120  # 최대 2분 대기
                wait_start = time.time()
                
                while time.time() - wait_start < max_wait_time:
                    if preload_done_file.exists():
                        logger.debug("모델 사전 로드가 다른 프로세스에서 완료됨")
                        return
                    time.sleep(0.5)
                
                logger.warning("모델 사전 로드 대기 시간 초과 - 계속 진행")
                
    except Exception as e:
        logger.warning(f"파일 잠금 사전 로드 실패: {e}")
        # 폴백: 잠금 없이 기본 로딩 시도
        logger.info("🔄 폴백 모드: 잠금 없이 모델 사전 로드")
        
        # Midm-2.0 (폴백)
        try:
            midm = get_midm_path()
            if midm.exists():
                from src.utils.korean_text_model import KoreanTextModel
                _ = KoreanTextModel(model_path=str(midm), n_ctx=1024, n_threads=int(os.getenv("LOCAL_LLM_THREADS", "2")))
                logger.info("✅ Midm-2.0 폴백 사전 로드 완료")
        except Exception as e:
            logger.warning(f"Midm-2.0 폴백 사전 로드 실패(무시): {e}")

        # A.X 멀티모달 모델 (폴백)
        try:
            ax_dir = get_ax_vl_dir()
            if ax_dir.exists():
                from src.utils.ax_multimodal import AXMultimodalModel
                _ = AXMultimodalModel(model_path=str(ax_dir), device="auto", max_memory_gb=4)
                logger.info("✅ A.X 멀티모달 폴백 사전 로드 완료")
        except Exception as e:
            logger.warning(f"A.X 폴백 사전 로드 실패(무시): {e}")
    
    finally:
        # 잠금 파일 정리 (선택적)
        try:
            if lock_file_path.exists():
                lock_file_path.unlink()
        except Exception:
            pass
