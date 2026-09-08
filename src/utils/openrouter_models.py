"""
OpenRouter 모델 리스트 조회 유틸리티

기능:
- OpenRouter /v1/models 엔드포인트에서 모델 목록 조회
- 간단한 TTL 캐싱으로 불필요한 호출 감소
"""

from __future__ import annotations

import time
import requests
from typing import List, Dict, Any

from config import settings


_cache: Dict[str, Any] = {
    "models": [],
    "ts": 0.0,
}


def _now() -> float:
    try:
        return time.perf_counter()
    except Exception:
        return time.time()


def list_openrouter_models(refresh: bool = False) -> List[str]:
    """
    OpenRouter 모델 ID 리스트를 반환

    - 캐싱: 환경변수 OPENROUTER_MODELS_CACHE_SECONDS(기본 600초)
    - 실패 시 빈 리스트 반환
    """
    cache_ttl = 600
    try:
        cache_ttl = int(getattr(settings, "openrouter_models_cache_seconds", 600))
    except Exception:
        pass

    now = _now()
    if (not refresh) and _cache["models"] and (now - float(_cache["ts"])) < cache_ttl:
        return list(_cache["models"])  # 복사본 반환

    base = getattr(settings, "openrouter_api_base", "https://openrouter.ai/api").rstrip("/")
    url = f"{base}/v1/models"
    headers = {"Content-Type": "application/json"}
    api_key = getattr(settings, "openrouter_api_key", None)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return list(_cache["models"]) or []
        data = r.json()
        items = data.get("data", [])
        # 각 item은 {id, name, ...} 형태가 일반적. id 우선 사용
        ids: List[str] = []
        for it in items:
            mid = it.get("id") or it.get("name")
            if isinstance(mid, str):
                ids.append(mid)
        ids.sort()
        _cache["models"], _cache["ts"] = ids, now
        return list(ids)
    except Exception:
        return list(_cache["models"]) or []


def search_openrouter_models(query: str, refresh: bool = False, limit: int = 200) -> List[str]:
    """
    간단한 포함 검색으로 모델 필터링
    """
    q = (query or "").strip().lower()
    models = list_openrouter_models(refresh=refresh)
    if not q:
        return models[:limit]
    filtered = [m for m in models if q in m.lower()]
    return filtered[:limit]
