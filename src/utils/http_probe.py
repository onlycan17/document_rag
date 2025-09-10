from __future__ import annotations

import requests
from typing import Dict, Optional


def probe_local_server(base: str, timeout: float = 3.0) -> Dict[str, Optional[bool]]:
    """
    로컬 OpenAI-호환 서버 상태 점검.
    - /health, /v1/models, /v1/queue/stats 간단 확인
    반환: {"health": bool|None, "models": bool|None, "queue": bool|None}
    """
    base = base.rstrip('/')
    result = {"health": None, "models": None, "queue": None}
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
        result["health"] = (r.status_code == 200)
    except Exception:
        result["health"] = False
    try:
        r = requests.get(f"{base}/v1/models", timeout=timeout)
        result["models"] = (r.status_code == 200)
    except Exception:
        result["models"] = False
    try:
        r = requests.get(f"{base}/v1/queue/stats", timeout=timeout)
        result["queue"] = (r.status_code == 200)
    except Exception:
        result["queue"] = False
    return result

