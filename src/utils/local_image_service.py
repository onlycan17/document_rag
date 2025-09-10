from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)


def _pick_base_url() -> str:
    """멀티 서버 설정에서 첫 번째 BASE를 선택(간단 전략)."""
    endpoints: List[str] = []
    if getattr(settings, "local_llm_base_urls", None):
        endpoints = [e.strip().rstrip('/') for e in str(settings.local_llm_base_urls).split(',') if e.strip()]
    base_default = getattr(settings, "local_llm_base_url", "").rstrip('/')
    if base_default and base_default not in endpoints:
        endpoints.append(base_default)
    # 1620 포트를 우선 선택
    prefer = getattr(settings, "local_mm_prefer_port", "1620")
    for e in endpoints:
        if f":{prefer}" in e:
            return e
    return endpoints[0] if endpoints else base_default


class LocalImageService:
    """
    로컬 OpenAI 호환 서버를 통한 이미지 처리/분석 헬퍼.
    - OCR 유사(비전 모델을 이용한 텍스트 추출): chat/completions 멀티모달 사용
    - 관련성 판정: score(0~1)와 설명 내도록 프롬프트 구성
    - (옵션) 향상/편집: /v1/images/edits 사용 가능
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base = (base_url or _pick_base_url()).rstrip('/')
        self.api_key = api_key or settings.local_llm_api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
        })

    def _image_to_data_url(self, image_path: str) -> str:
        with Image.open(image_path) as im:
            buf = BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

    def analyze_image(self, image_path: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        멀티모달 chat/completions로 OCR+관련성+설명을 한 번에 요청.
        응답 포맷(권장):
            RELEVANCE: <0~1>
            DESCRIPTION: <짧은 설명>
            TEXT: <검출 텍스트>
        """
        url = f"{self.base}/v1/chat/completions"
        image_url = self._image_to_data_url(image_path)
        prompt = (
            "당신은 문서 이미지 분석 보조자입니다.\n"
            "1) 이 이미지가 문서 이해에 얼마나 관련 있는지 0~1로 점수화하세요.\n"
            "2) 한 줄 설명을 주세요.\n"
            "3) 이미지 속 텍스트를 가능한 정확히 추출하세요.\n"
            "출력 형식:\nRELEVANCE: <0~1>\nDESCRIPTION: <text>\nTEXT: <text>"
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "이미지 분석과 OCR을 수행하는 도우미"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_image", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }
        r = self.session.post(url, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.debug(f"[local_image_service] raw_response: {text[:500]}")
        relevance = 0.0
        desc = ""
        extracted = ""
        # 단순 파싱
        try:
            for line in text.splitlines():
                if line.strip().lower().startswith("relevance:"):
                    relevance = float(line.split(":", 1)[1].strip())
                elif line.strip().lower().startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
                elif line.strip().lower().startswith("text:"):
                    extracted = line.split(":", 1)[1].strip()
        except Exception:
            pass
        return {"relevance": relevance, "description": desc, "text": extracted, "raw": text}

    def _get_queue_stats(self) -> Dict[str, Any]:
        try:
            url = f"{self.base}/v1/queue/stats"
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                return r.json() or {}
        except Exception:
            return {}
        return {}

    def _maybe_throttle(self, page_index: int, total_pages: int) -> None:
        stats = self._get_queue_stats()
        try:
            pending = int(stats.get("pending", 0))
            running = int(stats.get("running", 0))
            concurrency = int(stats.get("concurrency", 1)) or 1
        except Exception:
            pending = running = 0
            concurrency = 1
        # 단순 스로틀 규칙: pending이 동시성의 2배 초과 또는 running >= concurrency면 잠시 대기
        if pending > 2 * concurrency or running >= concurrency:
            import time
            logger.info(
                f"[local_image_service] throttling: pending={pending}, running={running}, concurrency={concurrency}"
            )
            time.sleep(0.5)

    def _enhance_image_if_enabled(self, image_path: Path) -> Path:
        if not settings.enable_image_enhancement:
            return image_path
        try:
            url = f"{self.base}/v1/images/edits"
            # 단순 JSON 바디 시도(서버 구현에 따라 다를 수 있어 실패 시 원본 유지)
            import base64
            with Image.open(image_path) as im:
                buf = BytesIO()
                im.convert("RGB").save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            body = {
                "prompt": settings.image_enhancement_prompt,
                "image": f"data:image/png;base64,{b64}",
            }
            r = self.session.post(url, json=body, timeout=60)
            if r.status_code == 200:
                data = r.json() or {}
                out_b64 = (data.get("data") or [{}])[0].get("b64_json")
                if out_b64:
                    out_bytes = base64.b64decode(out_b64)
                    enhanced_path = image_path.parent / ("enhanced_" + image_path.name)
                    with open(enhanced_path, "wb") as f:
                        f.write(out_bytes)
                    return enhanced_path
        except Exception as e:
            logger.debug(f"[local_image_service] enhancement failed, keep original: {e}")
        return image_path

    def process_pdf(self, pdf_path: str, output_dir: str, relevance_threshold: float, progress_callback=None) -> Dict[str, Any]:
        """
        PDF에서 이미지를 추출하고, 각 이미지를 analyze_image로 평가/OCR 수행.
        저장 규칙: relevance >= threshold 인 이미지만 저장.
        반환 스키마: 기존 지능형 추출기와 최대한 유사하게 구성.
        """
        out = Path(output_dir)
        images_dir = out / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        stats = {"total_images_found": 0, "relevant_images_saved": 0, "text_images_converted": 0}
        results: List[Dict[str, Any]] = []

        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            image_list = page.get_images(full=True)
            if progress_callback:
                progress_callback(0.1 + 0.6 * (page_index / max(1, len(doc))), f"이미지 분석 중... {page_index+1}/{len(doc)}")
            for img_idx, img in enumerate(image_list):
                # 큐 혼잡 시 스로틀
                self._maybe_throttle(page_index, len(doc))
                stats["total_images_found"] += 1
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_name = f"{Path(pdf_path).stem}_page{page_index+1}_img{img_idx+1}.png"
                temp_path = images_dir / ("tmp_" + img_name)
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)

                # 분석
                try:
                    # 선택적 이미지 향상 후 분석
                    use_path = self._enhance_image_if_enabled(temp_path)
                    info = self.analyze_image(str(use_path))
                except Exception as e:
                    logger.warning(f"이미지 분석 실패: {temp_path.name}: {e}")
                    temp_path.unlink(missing_ok=True)
                    continue

                keep = info.get("relevance", 0.0) >= relevance_threshold
                saved_path = None
                if keep:
                    saved_path = images_dir / img_name
                    os.replace(temp_path, saved_path)
                    stats["relevant_images_saved"] += 1
                else:
                    temp_path.unlink(missing_ok=True)

                extracted_text = info.get("text") or ""
                if extracted_text:
                    stats["text_images_converted"] += 1

                results.append({
                    "page": page_index + 1,
                    "image_file": str(saved_path or ""),
                    "saved": bool(keep),
                    "relevance_score": float(info.get("relevance", 0.0)),
                    "description": info.get("description", ""),
                    "extracted_text": extracted_text,
                })

        return {
            "images": results,
            "statistics": stats,
            "document_topic": {},
        }
