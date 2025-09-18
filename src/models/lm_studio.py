"""
LM Studio 통합 유틸리티

기능:
- LM Studio 로컬 서버(HTTP)에서 모델 목록을 조회
- HTTP이 실패하면 로컬 모델 디렉토리를 스캔하여 모델 목록 반환

환경변수:
- LM_STUDIO_API_URL: LM Studio 서버 기본 URL (기본: http://localhost:8080)
- LM_STUDIO_MODEL_DIR: LM Studio 모델 디렉토리 (기본: ~/Library/Application Support/lm-studio/models)

이 모듈은 사이드바나 초기화 코드에서 사용되어 사용자에게 로컬 모델 목록을 표시합니다.
"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any

from config import settings

logger = logging.getLogger(__name__)

import requests


def _get_env(name: str, default: str = None) -> str:
    return os.environ.get(name, default)


def list_models_via_api(base_url: str) -> List[Dict[str, Any]]:
    """LM Studio HTTP API로 모델 목록을 조회합니다.

    예상 엔드포인트: GET {base_url.rstrip('/')}/v1/models
    반환 형식은 내부적으로 변형하여 아래와 같은 리스트를 반환합니다:
    [{"id": "model-id", "name": "Model Name", "description": "..."}, ...]
    """
    def _candidate_urls(b: str) -> List[str]:
        bb = (b or '').rstrip('/')
        return [f"{bb}/v1/models", f"{bb}/models"]

    def _try_fetch_one(url: str):
        try:
            r = requests.get(url, timeout=3)
            r.raise_for_status()
            logger.info(f"LM Studio models fetched from {url}")
            return r.json()
        except Exception as e:
            logger.debug(f"LM Studio models fetch failed for {url}: {e}")
            return None

    data = None
    for url in _candidate_urls(base_url):
        data = _try_fetch_one(url)
        if data is not None:
            break

    if data is None:
        raise RuntimeError("No response from LM Studio model endpoints")

    models: List[Dict[str, Any]] = []
    # 다양한 서버 구현을 고려해 안전하게 파싱
    # LM Studio(및 OpenAI-like) 응답은 {'data': [ ... ]} 형식을 사용할 수 있음
    entries = []
    if isinstance(data, dict):
        # 우선 'models' 키, 그 다음 'data' 키를 시도
        entries = data.get('models') or data.get('data') or []
        if not isinstance(entries, list):
            entries = []
    elif isinstance(data, list):
        entries = data

    for m in entries:
        if not isinstance(m, dict):
            continue
        model_id = m.get('id') or m.get('model') or m.get('name')
        models.append({
            'id': model_id,
            'model': model_id,
            'name': m.get('name') or model_id,
            'description': m.get('description') or '',
            'meta': m
        })

    return models


def list_models_from_dir(model_dir: str) -> List[Dict[str, Any]]:
    """로컬 모델 디렉토리를 스캔하여 모델 파일 목록을 반환합니다.

    기본적으로 .bin, .gguf, .pt, .safetensors 같은 파일들을 모델로 간주합니다n+    """
    # Fix: remove stray characters in docstring and expand path
    p = Path(model_dir).expanduser()
    results: List[Dict[str, Any]] = []
    if not p.exists() or not p.is_dir():
        return results

    exts = {'.bin', '.gguf', '.pt', '.safetensors', '.pth'}

    # top-level files
    files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]
    for f in files:
        results.append({
            'id': f.stem,
            'model': f.stem,
            'name': f.name,
            'description': f'Local model file: {f.name}',
            'file_path': str(f)
        })

    # first matching child file inside each subdir
    dirs = [d for d in sorted(p.iterdir()) if d.is_dir()]
    for d in dirs:
        for child in d.iterdir():
            if child.is_file() and child.suffix.lower() in exts:
                results.append({
                    'id': d.name,
                    'model': d.name,
                    'name': child.name,
                    'description': f'Local model dir: {d.name}/{child.name}',
                    'file_path': str(child)
                })
                break

    return results


def list_lm_studio_models() -> Dict[str, List[Dict[str, Any]]]:
    """통합된 모델 목록 반환

    반환값 예시:
    {
        'local': [{...}, ...],
    }
    """
    models: Dict[str, List[Dict[str, Any]]] = {'local': []}

    def _gather_candidate_bases() -> List[str]:
        cands = []
        env_api = os.environ.get('LM_STUDIO_API_URL')
        if env_api:
            cands.append(env_api)
        cfg_api = getattr(settings, 'lm_studio_api_url', None)
        if cfg_api:
            cands.append(cfg_api)
        try:
            if getattr(settings, 'local_llm_base_url', None):
                cands.append(getattr(settings, 'local_llm_base_url'))
        except Exception:
            pass
        try:
            urls = getattr(settings, 'local_llm_base_urls', None)
            if urls:
                for u in str(urls).split(','):
                    if u.strip():
                        cands.append(u.strip())
        except Exception:
            pass
        # dedupe/normalize
        seen = set()
        out = []
        for c in cands:
            if not c:
                continue
            cc = c.strip()
            if cc and cc not in seen:
                seen.add(cc)
                out.append(cc)
        return out

    model_dir = _get_env('LM_STUDIO_MODEL_DIR', getattr(settings, 'lm_studio_model_dir', os.path.join(Path.home(), 'Library', 'Application Support', 'lm-studio', 'models')))

    candidate_bases = _gather_candidate_bases()

    # 1) 후보 base들에 대해 API 시도
    for base in candidate_bases:
        try:
            api_models = list_models_via_api(base)
            if api_models:
                for m in api_models:
                    m.setdefault('base_url', base)
                models['local'] = api_models
                return models
        except Exception as e:
            logger.debug(f"LM Studio discovery failed for {base}: {e}")
            continue

    # 2) 디렉토리 스캔
    try:
        dir_models = list_models_from_dir(model_dir)
        if dir_models:
            models['local'] = dir_models
            return models
    except Exception:
        pass

    return models
