from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import requests
from PIL import Image
import logging

from config import settings
logger = logging.getLogger(__name__)


class OpenRouterImageService:
    """
    OpenRouter 멀티모달(Chat Completions) 기반 이미지 분석/OCR 서비스.
    - 모델 기본값: qwen/qwen2.5-vl-32b-instruct
    - 엔드포인트: {base}/v1/chat/completions
    """

    def __init__(self):
        self.base = settings.openrouter_api_base.rstrip("/")
        self.key = settings.openrouter_api_key or ""
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.key}",
            "HTTP-Referer": "https://local",  # OpenRouter 권장 헤더(없어도 작동)
            "X-Title": "RAG-Pipeline-Preprocessing",
        })

    def _image_data_url(self, image_path: str) -> str:
        with Image.open(image_path) as im:
            buf = BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        url = f"{self.base}/v1/chat/completions"
        image_url = self._image_data_url(image_path)
        prompt = (
            "당신은 문서 이미지 분석 보조자입니다.\n"
            "1) 이 이미지가 문서 이해에 얼마나 관련 있는지 0~1로 점수화하세요.\n"
            "2) 한 줄 설명을 주세요.\n"
            "3) 이미지 속 텍스트를 가능한 정확히 추출하세요.\n"
            "출력 형식:\nRELEVANCE: <0~1>\nDESCRIPTION: <text>\nTEXT: <text>"
        )
        body = {
            "model": settings.openrouter_mm_model,
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
        r = self.session.post(url, json=body, timeout=90)
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.debug(f"[openrouter_image_service] raw_response: {text[:500]}")
        rel, desc, ocr = 0.0, "", ""
        for line in text.splitlines():
            low = line.strip().lower()
            if low.startswith("relevance:"):
                try:
                    rel = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif low.startswith("description:"):
                desc = line.split(":", 1)[1].strip()
            elif low.startswith("text:"):
                ocr = line.split(":", 1)[1].strip()
        return {"relevance": rel, "description": desc, "text": ocr, "raw": text}

    def process_pdf(self, pdf_path: str, output_dir: str, relevance_threshold: float) -> Dict[str, Any]:
        out = Path(output_dir)
        images_dir = out / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        stats = {"total_images_found": 0, "relevant_images_saved": 0, "text_images_converted": 0}
        results: List[Dict[str, Any]] = []

        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                stats["total_images_found"] += 1
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_name = f"{Path(pdf_path).stem}_page{page_index+1}_img{img_idx+1}.png"
                temp_path = images_dir / ("tmp_" + img_name)
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)

                info = self.analyze_image(str(temp_path))
                keep = info.get("relevance", 0.0) >= relevance_threshold
                saved_path = None
                if keep:
                    saved_path = images_dir / img_name
                    temp_path.replace(saved_path)
                    stats["relevant_images_saved"] += 1
                else:
                    temp_path.unlink(missing_ok=True)

                if info.get("text"):
                    stats["text_images_converted"] += 1

                results.append({
                    "page": page_index + 1,
                    "image_file": str(saved_path or ""),
                    "saved": bool(keep),
                    "relevance_score": float(info.get("relevance", 0.0)),
                    "description": info.get("description", ""),
                    "extracted_text": info.get("text", ""),
                })
        return {"images": results, "statistics": stats, "document_topic": {}}
