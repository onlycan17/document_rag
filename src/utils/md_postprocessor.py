"""
MD 후처리 엔진 - 원본 MD 파일을 LLM으로 정제하여 완벽한 문맥 연결 MD 파일 생성
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from config import settings
from src.agents.base_agent import BaseAgent
from src.processing.chunked_cleanup import MAX_OUTPUT_TOKENS, clean_document

logger = logging.getLogger(__name__)


class MDPostProcessor(BaseAgent):
    """
    MD 파일 후처리 전문 클래스
    원본 MD → 문맥 연결된 고품질 MD 변환
    """

    def __init__(
        self,
        output_dir: str = "processed_docs",
        target_quality: int = 90,
        provider: str | None = None,
        model_name: str | None = None,
    ):
        # 선택한 제공자/모델을 그대로 사용(없으면 설정/세션)
        if provider is None:
            try:
                import streamlit as st  # type: ignore

                provider = st.session_state.get("current_provider", None)
                model_name = model_name or st.session_state.get("current_model", None)
            except Exception as err:
                logger.debug(f"세션 provider/model 조회 실패(무시): {err}")
        super().__init__("MDPostProcessor", provider=provider, model_name=model_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_quality = target_quality
        self.last_quality_score = 0.0
        self.quality_checker = None  # 품질 검사기 (옵션)

        # 처리 통계
        self.stats = {
            "total_chunks": 0,
            "processed_chunks": 0,
            "fallback_chunks": [],
            "total_chars": 0,
            "processing_time": 0.0,
        }

        logger.info(f"🔧 MD 후처리 엔진 초기화 완료: {self.output_dir}")

    def set_quality_checker(self, quality_checker):
        """품질 검사기 설정"""
        self.quality_checker = quality_checker
        logger.info("품질 검사기 연결됨")

    def process(self, input_data: Any) -> Any:
        """
        에이전트 베이스 클래스의 추상 메서드 구현
        MD 파일 경로를 받아 처리된 파일 경로 반환
        """
        if isinstance(input_data, str):
            return self.process_md_file(input_data)
        else:
            raise ValueError("입력 데이터는 MD 파일 경로(문자열)여야 합니다")

    def process_file(self, md_file_path: str, progress_callback: Optional[Callable] = None) -> str:
        """
        MD 파일을 처리하는 메인 엔트리 포인트
        """
        return self.process_md_file(md_file_path, progress_callback)

    def process_md_file(self, md_file_path: str, progress_callback: Optional[Callable] = None) -> str:
        """
        MD 파일을 LLM으로 후처리하여 완벽한 문맥 연결 수행

        Args:
            md_file_path: 원본 MD 파일 경로
            progress_callback: 진행 상황 콜백 함수 (progress: float, message: str)

        Returns:
            처리된 MD 파일 경로
        """
        start_time = time.time()
        md_path = Path(md_file_path)

        if not md_path.exists():
            raise FileNotFoundError(f"MD 파일을 찾을 수 없습니다: {md_file_path}")

        logger.info(f"📝 MD 후처리 시작: {md_path.name}")

        if progress_callback:
            progress_callback(0.1, "MD 파일 읽기...")

        # 1단계: 원본 MD 파일 읽기
        with open(md_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        logger.info(f"📊 원본 텍스트: {len(original_content):,}자")

        if progress_callback:
            progress_callback(0.2, "텍스트 청킹...")

        # 2~4단계: 문장 경계 분할 → 앞뒤 문맥을 붙여 병렬 교정 → 길이 검증 실패 청크는 원문 유지
        result = clean_document(
            original_content,
            lambda prompt: self._call_llm(prompt, temperature=0.1, max_tokens=MAX_OUTPUT_TOKENS),
            max_workers=settings.parallel_workers,
        )
        final_content = result.text
        self.stats["total_chunks"] = result.total_chunks
        self.stats["processed_chunks"] = result.total_chunks - len(result.fallback_chunks)
        self.stats["fallback_chunks"] = result.fallback_chunks
        self.stats["total_chars"] = len(original_content)

        if progress_callback:
            progress_callback(
                0.9, f"{result.total_chunks}개 청크 교정 완료 (원문 유지 {len(result.fallback_chunks)}개)"
            )

        # 5단계: 결과 저장
        output_filename = f"{md_path.stem}_processed.md"
        output_path = self.output_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        # 통계 업데이트
        self.stats["processing_time"] = time.time() - start_time

        # 품질 점수 계산 (옵션)
        if self.quality_checker:
            try:
                quality_result = self.quality_checker.analyze_quality(str(output_path))
                self.last_quality_score = quality_result.get("total_score", 0.0)
                logger.info(f"📊 품질 점수: {self.last_quality_score:.1f}점")
            except Exception as e:
                logger.warning(f"품질 검사 실패: {str(e)}")
                self.last_quality_score = 0.0

        if progress_callback:
            progress_callback(1.0, f"완료: {output_path.name}")

        logger.info(f"✅ MD 후처리 완료: {output_path}")
        logger.info(
            f"📊 처리 통계: {self.stats['processed_chunks']}/{self.stats['total_chunks']} 청크, "
            f"{self.stats['processing_time']:.1f}초"
        )

        return str(output_path)

    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return self.stats.copy()

    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            "total_chunks": 0,
            "processed_chunks": 0,
            "fallback_chunks": [],
            "total_chars": 0,
            "processing_time": 0.0,
        }
