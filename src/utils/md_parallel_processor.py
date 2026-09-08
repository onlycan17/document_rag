"""
병렬 처리 관리자 - 여러 MD 파일을 동시에 후처리하고 품질 검증 수행
"""

import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from .md_postprocessor import MDPostProcessor
from .quality_checker import QualityChecker

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """처리 상태 열거형"""

    PENDING = "대기중"
    PROCESSING = "처리중"
    QUALITY_CHECK = "품질검증중"
    REFINEMENT = "재처리중"
    COMPLETED = "완료"
    FAILED = "실패"


@dataclass
class ProcessingTask:
    """처리 작업 단위"""

    input_path: str
    output_path: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    current_message: str = ""
    iterations: int = 0
    max_iterations: int = 3
    quality_score: float = 0.0
    target_score: float = 90.0
    error_message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def get_duration(self) -> float:
        """작업 소요 시간 반환"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


@dataclass
class ProcessingStats:
    """전체 처리 통계"""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_processing_time: float = 0.0
    average_quality_score: float = 0.0
    total_iterations: int = 0

    def get_success_rate(self) -> float:
        """성공률 반환"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100


class ParallelProcessor:
    """
    병렬 MD 후처리 관리자
    여러 문서를 동시에 처리하고 품질 기준에 도달할 때까지 반복 처리
    """

    def __init__(self, max_workers: int = 3, output_dir: str = "processed_docs"):
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 처리 컴포넌트(현재 선택된 제공자/모델 반영)
        _prov = None
        _model = None
        try:
            from config import settings as _settings

            try:
                import streamlit as st  # type: ignore

                _prov = st.session_state.get("current_provider", None)
                _model = st.session_state.get("current_model", None)
            except Exception:
                pass
            # 후처리 전용 오버라이드 우선
            _md_override_provider = getattr(_settings, "md_postprocess_provider", None)
            _md_override_model = getattr(_settings, "md_postprocess_model", None)
            if _md_override_provider:
                _prov = _md_override_provider
                if _md_override_model:
                    _model = _md_override_model
            _prov = _prov or getattr(_settings, "llm_provider", "local")
            if not _model:
                if _prov == "openai":
                    _model = getattr(_settings, "openai_model", None)
                elif _prov == "google":
                    _model = getattr(_settings, "google_model", None)
                elif _prov == "anthropic":
                    _model = getattr(_settings, "anthropic_model", None)
                elif _prov == "openrouter":
                    _model = getattr(_settings, "openrouter_model", None) or getattr(
                        _settings, "openrouter_mm_model", None
                    )
                else:
                    _model = None
        except Exception:
            pass
        self.md_processor = MDPostProcessor(str(self.output_dir), provider=_prov, model_name=_model)
        self.quality_checker = QualityChecker(provider=_prov, model_name=_model)

        # 상태 관리
        self.tasks: Dict[str, ProcessingTask] = {}
        self.stats = ProcessingStats()
        self._lock = threading.Lock()
        self._is_processing = False

        # 콜백 함수들
        self.progress_callback: Optional[Callable[[str, ProcessingTask], None]] = None
        self.completion_callback: Optional[Callable[[ProcessingStats], None]] = None

        logger.info(f"🚀 병렬 처리 관리자 초기화 완료: {max_workers}개 워커, 출력 디렉토리: {self.output_dir}")

    def add_file(self, input_path: str, max_iterations: int = 3, target_score: float = 90.0) -> str:
        """
        처리할 파일 추가

        Args:
            input_path: 입력 MD 파일 경로
            max_iterations: 최대 반복 횟수
            target_score: 목표 품질 점수

        Returns:
            작업 ID (파일 경로 기반)
        """
        input_path = str(Path(input_path).resolve())

        if not Path(input_path).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")

        task_id = input_path

        with self._lock:
            if task_id in self.tasks:
                logger.warning(f"⚠️ 이미 등록된 파일: {input_path}")
                return task_id

            self.tasks[task_id] = ProcessingTask(
                input_path=input_path, max_iterations=max_iterations, target_score=target_score
            )
            self.stats.total_tasks += 1

        logger.info(f"📝 파일 추가: {Path(input_path).name} (목표: {target_score}점, 최대 {max_iterations}회)")
        return task_id

    def add_files(self, input_paths: List[str], max_iterations: int = 3, target_score: float = 90.0) -> List[str]:
        """
        여러 파일을 일괄 추가

        Args:
            input_paths: 입력 MD 파일 경로 목록
            max_iterations: 최대 반복 횟수
            target_score: 목표 품질 점수

        Returns:
            작업 ID 목록
        """
        task_ids = []
        for path in input_paths:
            try:
                task_id = self.add_file(path, max_iterations, target_score)
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"❌ 파일 추가 실패: {path} - {str(e)}")
                task_ids.append(None)

        return task_ids

    def set_progress_callback(self, callback: Callable[[str, ProcessingTask], None]):
        """진행 상황 콜백 설정"""
        self.progress_callback = callback

    def set_completion_callback(self, callback: Callable[[ProcessingStats], None]):
        """완료 콜백 설정"""
        self.completion_callback = callback

    def start_processing(self) -> ProcessingStats:
        """
        모든 등록된 파일의 병렬 처리 시작

        Returns:
            처리 완료 후 통계
        """
        if self._is_processing:
            raise RuntimeError("이미 처리가 진행 중입니다")

        with self._lock:
            if not self.tasks:
                raise ValueError("처리할 파일이 없습니다")

            pending_tasks = [task for task in self.tasks.values() if task.status == ProcessingStatus.PENDING]
            if not pending_tasks:
                logger.warning("⚠️ 처리할 대기 중인 작업이 없습니다")
                return self.stats

        self._is_processing = True

        try:
            logger.info(f"🚀 병렬 처리 시작: {len(pending_tasks)}개 파일, {self.max_workers}개 워커")
            start_time = time.time()

            # ThreadPoolExecutor로 병렬 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 모든 작업 제출
                future_to_task = {}
                for task_id, task in self.tasks.items():
                    if task.status == ProcessingStatus.PENDING:
                        future = executor.submit(self._process_single_file, task_id, task)
                        future_to_task[future] = (task_id, task)

                # 완료된 작업 처리
                for future in as_completed(future_to_task):
                    task_id, task = future_to_task[future]
                    try:
                        future.result()
                        logger.info(f"✅ 작업 완료: {Path(task.input_path).name}")
                    except Exception as e:
                        logger.error(f"❌ 작업 실패: {Path(task.input_path).name} - {str(e)}")
                        with self._lock:
                            task.status = ProcessingStatus.FAILED
                            task.error_message = str(e)
                            self.stats.failed_tasks += 1

            # 최종 통계 계산
            self._calculate_final_stats()
            self.stats.total_processing_time = time.time() - start_time

            logger.info("🎉 병렬 처리 완료!")
            logger.info(
                f"📊 성공: {self.stats.completed_tasks}/{self.stats.total_tasks}개 "
                f"({self.stats.get_success_rate():.1f}%)"
            )
            logger.info(f"⏱️ 총 소요시간: {self.stats.total_processing_time:.1f}초")
            logger.info(f"📈 평균 품질점수: {self.stats.average_quality_score:.1f}점")

            # 완료 콜백 실행
            if self.completion_callback:
                try:
                    self.completion_callback(self.stats)
                except Exception as e:
                    logger.warning(f"⚠️ 완료 콜백 실행 실패: {str(e)}")

            return self.stats

        finally:
            self._is_processing = False

    def _process_single_file(self, task_id: str, task: ProcessingTask) -> bool:
        """
        단일 파일 처리 (반복 품질 검증 포함)

        Returns:
            처리 성공 여부
        """
        with self._lock:
            task.status = ProcessingStatus.PROCESSING
            task.start_time = time.time()
            self._notify_progress(task_id, task)

        try:
            current_file = task.input_path

            # 최대 반복 횟수까지 처리 및 품질 검증
            for iteration in range(1, task.max_iterations + 1):
                with self._lock:
                    task.iterations = iteration
                    task.current_message = f"반복 {iteration}/{task.max_iterations} 처리 중..."
                    self._notify_progress(task_id, task)

                # MD 후처리 실행
                try:
                    processed_file = self.md_processor.process_md_file(
                        current_file,
                        progress_callback=lambda p, m: self._update_task_progress(task_id, p * 0.7, f"처리 중: {m}"),
                    )

                    with self._lock:
                        task.status = ProcessingStatus.QUALITY_CHECK
                        task.current_message = f"품질 검증 중... ({iteration}회차)"
                        self._notify_progress(task_id, task)

                    # 품질 검증
                    quality_result = self.quality_checker.analyze_quality(processed_file)

                    with self._lock:
                        task.quality_score = quality_result["total_score"]
                        task.progress = 0.7 + (0.3 * (iteration / task.max_iterations))

                    logger.info(f"📊 {Path(task.input_path).name} - {iteration}회차 품질: {task.quality_score:.1f}점")

                    # 품질 기준 달성 확인
                    if quality_result["passed"]:
                        with self._lock:
                            task.status = ProcessingStatus.COMPLETED
                            task.output_path = processed_file
                            task.end_time = time.time()
                            task.progress = 1.0
                            task.current_message = f"완료! 품질점수: {task.quality_score:.1f}점"
                            self.stats.completed_tasks += 1
                            self.stats.total_iterations += iteration
                            self._notify_progress(task_id, task)

                        logger.info(
                            f"✅ {Path(task.input_path).name} 처리 완료: {task.quality_score:.1f}점 "
                            f"({iteration}회 반복)"
                        )
                        return True

                    # 품질 기준 미달성 시 다음 반복을 위해 파일 업데이트
                    current_file = processed_file

                    if iteration < task.max_iterations:
                        with self._lock:
                            task.status = ProcessingStatus.REFINEMENT
                            task.current_message = f"품질 개선 필요 ({task.quality_score:.1f}점), 재처리 준비..."
                            self._notify_progress(task_id, task)

                        logger.info(
                            f"🔄 {Path(task.input_path).name} 재처리 필요: {task.quality_score:.1f}점 "
                            f"< {task.target_score}점"
                        )

                except Exception as e:
                    logger.error(f"❌ {Path(task.input_path).name} 처리 실패 ({iteration}회차): {str(e)}")
                    raise

            # 최대 반복 횟수 도달 시 현재 결과로 완료 처리
            with self._lock:
                task.status = ProcessingStatus.COMPLETED
                task.output_path = current_file
                task.end_time = time.time()
                task.progress = 1.0
                task.current_message = f"최대 반복 완료 (품질점수: {task.quality_score:.1f}점)"
                self.stats.completed_tasks += 1
                self.stats.total_iterations += task.max_iterations
                self._notify_progress(task_id, task)

            logger.warning(
                f"⚠️ {Path(task.input_path).name} 최대 반복 도달: {task.quality_score:.1f}점 "
                f"(목표: {task.target_score}점)"
            )
            return True

        except Exception as e:
            with self._lock:
                task.status = ProcessingStatus.FAILED
                task.error_message = str(e)
                task.end_time = time.time()
                task.current_message = f"처리 실패: {str(e)}"
                self.stats.failed_tasks += 1
                self._notify_progress(task_id, task)

            logger.error(f"❌ {Path(task.input_path).name} 처리 실패: {str(e)}")
            return False

    def _update_task_progress(self, task_id: str, progress: float, message: str):
        """작업 진행 상황 업데이트"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.progress = min(1.0, max(0.0, progress))
                task.current_message = message
                self._notify_progress(task_id, task)

    def _notify_progress(self, task_id: str, task: ProcessingTask):
        """진행 상황 콜백 호출 (락 내부에서 호출됨)"""
        if self.progress_callback:
            try:
                self.progress_callback(task_id, task)
            except Exception as e:
                logger.warning(f"⚠️ 진행 상황 콜백 실행 실패: {str(e)}")

    def _calculate_final_stats(self):
        """최종 통계 계산"""
        with self._lock:
            completed_tasks = [task for task in self.tasks.values() if task.status == ProcessingStatus.COMPLETED]

            if completed_tasks:
                total_quality = sum(task.quality_score for task in completed_tasks)
                self.stats.average_quality_score = total_quality / len(completed_tasks)

            self.stats.total_iterations = sum(task.iterations for task in self.tasks.values())

    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """특정 작업의 상태 조회"""
        with self._lock:
            return self.tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, ProcessingTask]:
        """모든 작업 상태 조회"""
        with self._lock:
            return self.tasks.copy()

    def get_stats(self) -> ProcessingStats:
        """처리 통계 조회"""
        with self._lock:
            return self.stats

    def is_processing(self) -> bool:
        """처리 진행 중인지 확인"""
        return self._is_processing

    def cancel_processing(self):
        """처리 취소 (현재 진행 중인 작업은 완료될 때까지 대기)"""
        logger.warning("⚠️ 처리 취소는 현재 구현되지 않았습니다. 진행 중인 작업이 완료될 때까지 대기해주세요.")

    def reset(self):
        """모든 작업과 통계 초기화"""
        if self._is_processing:
            raise RuntimeError("처리 진행 중에는 초기화할 수 없습니다")

        with self._lock:
            self.tasks.clear()
            self.stats = ProcessingStats()

        logger.info("🔄 병렬 처리 관리자 초기화 완료")
