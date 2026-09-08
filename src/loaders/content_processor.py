"""
콘텐츠 처리 모듈

이 모듈은 기존 document_loader.py에서 전처리 오케스트레이션 및 에이전트 기반 처리 로직을 분리하여
단일 책임 원칙에 따라 독립적으로 관리합니다.

주요 기능:
- 에이전트 기반 전처리 오케스트레이션
- 멀티모달 처리 통합
- 이미지 분석 및 OCR 처리
- 품질 검증 및 후처리
- 병렬 처리 및 성능 최적화
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import Document

# 에이전트 및 처리 모듈 import (현재 구현체 이름과 호환되도록 별칭 부여)
try:
    from src.agents.base_agent import BaseAgent
    from src.agents.context_connector import ContextConnectorAgent as ContextConnector
    from src.agents.quality_validator import QualityValidatorAgent as QualityValidator
    from src.agents.structure_parser import StructureParserAgent as StructureParser
    from src.utils.agent_pdf_converter import AgentBasedPDFConverter as AgentPDFConverter
    from src.utils.openrouter_image_service import OpenRouterImageService
except ImportError as e:
    logging.warning(f"일부 에이전트 모듈을 임포트할 수 없습니다: {e}")
    BaseAgent = None
    ContextConnector = None
    QualityValidator = None
    StructureParser = None
    AgentPDFConverter = None
    OpenRouterImageService = None

# 설정값 import
try:
    from config import settings
except ImportError:

    class DefaultSettings:
        enable_agent_preprocessing = True
        enable_multimodal_preprocessing = True
        enable_parallel_processing = True
        max_workers = 3
        processing_timeout = 300
        image_analysis_provider = "openrouter"
        openrouter_mm_model = "z-ai/glm-4.5v"

    settings = DefaultSettings()

logger = logging.getLogger(__name__)


class ContentProcessor:
    """
    콘텐츠 처리 및 전처리 오케스트레이션 클래스

    기존 document_loader.py에서 에이전트 기반 처리 로직을 분리하여
    확장 가능하고 모듈화된 전처리 시스템으로 구성
    """

    def __init__(self):
        """ContentProcessor 초기화"""
        # 설정값 로드
        self.enable_agent_preprocessing = getattr(settings, "enable_agent_preprocessing", True)
        self.enable_multimodal_preprocessing = getattr(settings, "enable_multimodal_preprocessing", True)
        self.enable_parallel_processing = getattr(settings, "enable_parallel_processing", True)
        self.max_workers = getattr(settings, "max_workers", 3)
        self.processing_timeout = getattr(settings, "processing_timeout", 300)

        # 에이전트 초기화
        self.agents = self._init_agents()

        # 멀티모달 서비스 초기화
        self.multimodal_services = self._init_multimodal_services()

        # 처리 파이프라인 초기화
        self.processing_pipeline = self._init_processing_pipeline()

        # 품질 검증기 초기화
        self.quality_validator = self._init_quality_validator()

        logger.info("ContentProcessor 초기화 완료")

    def _init_agents(self) -> Dict[str, Any]:
        """에이전트 모듈들 초기화"""
        agents = {}

        try:
            if BaseAgent:
                agents["base"] = BaseAgent()

            if ContextConnector:
                agents["context_connector"] = ContextConnector()

            if StructureParser:
                agents["structure_parser"] = StructureParser()

            if AgentPDFConverter:
                agents["pdf_converter"] = AgentPDFConverter()

            logger.info(f"에이전트 초기화 완료: {list(agents.keys())}")

        except Exception as e:
            logger.error(f"에이전트 초기화 중 오류: {str(e)}")

        return agents

    def _init_multimodal_services(self) -> Dict[str, Any]:
        """멀티모달 서비스 초기화"""
        services = {}

        try:
            if OpenRouterImageService:
                services["openrouter"] = OpenRouterImageService()
                logger.info("OpenRouter 이미지 서비스 초기화 완료")

        except Exception as e:
            logger.error(f"멀티모달 서비스 초기화 중 오류: {str(e)}")

        return services

    def _init_processing_pipeline(self) -> List[Callable]:
        """처리 파이프라인 초기화"""
        pipeline = []

        # 기본 파이프라인 단계들
        pipeline.extend(
            [self._preprocess_content, self._extract_structure, self._enhance_metadata, self._validate_quality]
        )

        # 멀티모달 처리 단계 (활성화된 경우)
        if self.enable_multimodal_preprocessing:
            pipeline.insert(-1, self._process_multimodal_content)

        # 에이전트 처리 단계 (활성화된 경우)
        if self.enable_agent_preprocessing:
            pipeline.insert(-1, self._process_with_agents)

        return pipeline

    def _init_quality_validator(self):
        """품질 검증기 초기화"""
        try:
            if QualityValidator:
                return QualityValidator()
        except Exception as e:
            logger.error(f"품질 검증기 초기화 중 오류: {str(e)}")

        return None

    def process_documents(
        self, documents: List[Document], file_path: Optional[str] = None, progress_callback: Optional[Callable] = None
    ) -> List[Document]:
        """
        문서 리스트 전처리

        Args:
            documents: 처리할 문서 리스트
            file_path: 원본 파일 경로 (멀티모달 처리용)
            progress_callback: 진행 상황 콜백

        Returns:
            전처리된 문서 리스트
        """
        if not documents:
            return []

        try:
            logger.info(f"문서 전처리 시작: {len(documents)} 문서")

            # 병렬 처리 여부에 따라 분기
            if self.enable_parallel_processing and len(documents) > 1:
                processed_docs = self._process_documents_parallel(documents, file_path, progress_callback)
            else:
                processed_docs = self._process_documents_sequential(documents, file_path, progress_callback)

            logger.info(f"문서 전처리 완료: {len(processed_docs)} 문서")
            return processed_docs

        except Exception as e:
            logger.error(f"문서 전처리 중 오류: {str(e)}")
            return documents

    def _process_documents_parallel(
        self, documents: List[Document], file_path: Optional[str] = None, progress_callback: Optional[Callable] = None
    ) -> List[Document]:
        """병렬 문서 처리"""
        try:
            processed_docs = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 문서들을 병렬로 처리
                future_to_doc = {
                    executor.submit(self._process_single_document, doc, file_path, i, len(documents)): doc
                    for i, doc in enumerate(documents)
                }

                for future in as_completed(future_to_doc, timeout=self.processing_timeout):
                    try:
                        result = future.result()
                        if result:
                            processed_docs.extend(result)

                        if progress_callback:
                            progress_callback(len(processed_docs), len(documents))

                    except Exception as e:
                        original_doc = future_to_doc[future]
                        logger.error(f"문서 처리 실패: {str(e)}")
                        processed_docs.append(original_doc)  # 원본 유지

            return processed_docs

        except Exception as e:
            logger.error(f"병렬 처리 중 오류: {str(e)}")
            return self._process_documents_sequential(documents, file_path, progress_callback)

    def _process_documents_sequential(
        self, documents: List[Document], file_path: Optional[str] = None, progress_callback: Optional[Callable] = None
    ) -> List[Document]:
        """순차 문서 처리"""
        try:
            processed_docs = []

            for i, doc in enumerate(documents):
                try:
                    result = self._process_single_document(doc, file_path, i, len(documents))
                    if result:
                        processed_docs.extend(result)

                    if progress_callback:
                        progress_callback(i + 1, len(documents))

                except Exception as e:
                    logger.error(f"문서 {i} 처리 실패: {str(e)}")
                    processed_docs.append(doc)  # 원본 유지

            return processed_docs

        except Exception as e:
            logger.error(f"순차 처리 중 오류: {str(e)}")
            return documents

    def _process_single_document(
        self, document: Document, file_path: Optional[str] = None, doc_index: int = 0, total_docs: int = 1
    ) -> List[Document]:
        """단일 문서 처리"""
        try:
            # 처리 컨텍스트 생성
            context = {
                "document": document,
                "file_path": file_path,
                "doc_index": doc_index,
                "total_docs": total_docs,
                "metadata": document.metadata.copy() if document.metadata else {},
            }

            # 파이프라인 단계별 처리
            for step in self.processing_pipeline:
                try:
                    context = step(context)
                    if not context or not context.get("document"):
                        logger.warning(f"파이프라인 단계에서 문서가 소실됨: {step.__name__}")
                        return [document]
                except Exception as e:
                    logger.error(f"파이프라인 단계 '{step.__name__}' 실패: {str(e)}")
                    continue

            # 결과 반환
            result_doc = context.get("document")
            if isinstance(result_doc, list):
                return result_doc
            else:
                return [result_doc] if result_doc else [document]

        except Exception as e:
            logger.error(f"단일 문서 처리 중 오류: {str(e)}")
            return [document]

    def _preprocess_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """기본 콘텐츠 전처리"""
        try:
            document = context["document"]
            content = document.page_content

            # 기본 텍스트 정제
            if content:
                # 불필요한 공백 정리
                content = " ".join(content.split())

                # 메타데이터 정보 추가
                context["metadata"].update(
                    {
                        "original_length": len(document.page_content),
                        "processed_length": len(content),
                        "preprocessing_applied": True,
                    }
                )

                # 문서 업데이트
                context["document"] = Document(page_content=content, metadata=context["metadata"])

            return context

        except Exception as e:
            logger.error(f"기본 전처리 중 오류: {str(e)}")
            return context

    def _extract_structure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """구조 정보 추출"""
        try:
            if "structure_parser" in self.agents:
                parser = self.agents["structure_parser"]
                document = context["document"]

                # 구조 정보 추출
                structure_info = parser.parse_structure(document.page_content)

                if structure_info:
                    context["metadata"]["structure_info"] = structure_info
                    context["metadata"]["has_structure"] = True

            return context

        except Exception as e:
            logger.error(f"구조 추출 중 오류: {str(e)}")
            return context

    def _process_multimodal_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """멀티모달 콘텐츠 처리"""
        try:
            file_path = context.get("file_path")

            # PDF 파일이고 이미지 서비스가 있는 경우
            if file_path and Path(file_path).suffix.lower() == ".pdf" and "openrouter" in self.multimodal_services:
                service = self.multimodal_services["openrouter"]

                # 이미지 분석 수행
                image_analysis = service.analyze_document_images(file_path)

                if image_analysis:
                    context["metadata"]["image_analysis"] = image_analysis
                    context["metadata"]["has_images"] = True

                    # 이미지 정보를 텍스트에 통합
                    document = context["document"]
                    enhanced_content = self._integrate_image_analysis(document.page_content, image_analysis)

                    context["document"] = Document(page_content=enhanced_content, metadata=context["metadata"])

            return context

        except Exception as e:
            logger.error(f"멀티모달 처리 중 오류: {str(e)}")
            return context

    def _process_with_agents(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 기반 처리"""
        try:
            # PDF 변환 에이전트 사용
            if (
                "pdf_converter" in self.agents
                and context.get("file_path")
                and Path(context["file_path"]).suffix.lower() == ".pdf"
            ):
                converter = self.agents["pdf_converter"]
                enhanced_docs = converter.process_document(context["document"])

                if enhanced_docs and len(enhanced_docs) > 0:
                    # 여러 문서로 분할된 경우
                    if len(enhanced_docs) > 1:
                        context["document"] = enhanced_docs
                        return context
                    else:
                        context["document"] = enhanced_docs[0]

            # 컨텍스트 연결 에이전트 사용
            if "context_connector" in self.agents:
                connector = self.agents["context_connector"]
                connected_doc = connector.enhance_context(context["document"])

                if connected_doc:
                    context["document"] = connected_doc

            return context

        except Exception as e:
            logger.error(f"에이전트 처리 중 오류: {str(e)}")
            return context

    def _enhance_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 강화"""
        try:
            document = context["document"]
            metadata = context["metadata"]

            # 처리 통계 추가
            metadata.update(
                {
                    "processing_timestamp": logger.getEffectiveLevel(),  # 간단한 타임스탬프
                    "content_type": self._determine_content_type(document.page_content),
                    "language": self._detect_language(document.page_content),
                    "complexity_score": self._calculate_complexity(document.page_content),
                }
            )

            # 문서 업데이트
            if isinstance(document, list):
                for doc in document:
                    doc.metadata.update(metadata)
            else:
                document.metadata.update(metadata)
                context["document"] = document

            return context

        except Exception as e:
            logger.error(f"메타데이터 강화 중 오류: {str(e)}")
            return context

    def _validate_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """품질 검증"""
        try:
            if self.quality_validator:
                document = context["document"]

                if isinstance(document, list):
                    validated_docs = []
                    for doc in document:
                        if self.quality_validator.validate_document(doc):
                            validated_docs.append(doc)
                    context["document"] = validated_docs
                else:
                    if not self.quality_validator.validate_document(document):
                        logger.warning("문서 품질 검증 실패")
                        # 품질이 낮아도 문서는 유지 (원본 반환)

            return context

        except Exception as e:
            logger.error(f"품질 검증 중 오류: {str(e)}")
            return context

    def _integrate_image_analysis(self, text_content: str, image_analysis: Dict) -> str:
        """이미지 분석 결과를 텍스트에 통합"""
        try:
            enhanced_content = text_content

            # 이미지 정보가 있는 경우
            if image_analysis.get("images"):
                image_descriptions = []

                for img_info in image_analysis["images"]:
                    if img_info.get("description"):
                        image_descriptions.append(f"[이미지: {img_info['description']}]")

                if image_descriptions:
                    enhanced_content += "\n\n" + "\n".join(image_descriptions)

            # OCR 결과가 있는 경우
            if image_analysis.get("ocr_text"):
                enhanced_content += f"\n\n[OCR 추출 텍스트]\n{image_analysis['ocr_text']}"

            return enhanced_content

        except Exception as e:
            logger.error(f"이미지 분석 통합 중 오류: {str(e)}")
            return text_content

    def _determine_content_type(self, content: str) -> str:
        """콘텐츠 타입 결정"""
        try:
            if not content:
                return "empty"

            # 마크다운 검사
            if any(pattern in content for pattern in ["#", "```", "|"]):
                return "markdown"

            # 구조화된 문서 검사
            if any(pattern in content for pattern in ["제1장", "1.", "가.", "(1)"]):
                return "structured"

            # 일반 텍스트
            return "text"

        except Exception as e:
            logger.error(f"콘텐츠 타입 결정 중 오류: {str(e)}")
            return "unknown"

    def _detect_language(self, content: str) -> str:
        """언어 감지 (간단한 버전)"""
        try:
            if not content:
                return "unknown"

            # 한국어 비율 계산
            korean_chars = len([c for c in content if "가" <= c <= "힣"])
            total_chars = len([c for c in content if c.isalpha()])

            if total_chars == 0:
                return "unknown"

            korean_ratio = korean_chars / total_chars

            if korean_ratio > 0.3:
                return "korean"
            elif korean_ratio > 0.1:
                return "mixed"
            else:
                return "english"

        except Exception as e:
            logger.error(f"언어 감지 중 오류: {str(e)}")
            return "unknown"

    def _calculate_complexity(self, content: str) -> float:
        """콘텐츠 복잡도 점수 계산"""
        try:
            if not content:
                return 0.0

            score = 0.0

            # 길이 기반 점수
            length_score = min(len(content) / 1000, 5.0)  # 최대 5점
            score += length_score

            # 구조 복잡도
            structure_patterns = ["#", "1.", "가.", "(1)", "|", "```"]
            structure_score = sum(1 for pattern in structure_patterns if pattern in content)
            score += min(structure_score, 3.0)  # 최대 3점

            # 어휘 다양성
            words = content.split()
            unique_words = len(set(words))
            vocab_score = min(unique_words / len(words) if words else 0, 2.0) * 2  # 최대 2점
            score += vocab_score

            return min(score, 10.0)  # 최대 10점

        except Exception as e:
            logger.error(f"복잡도 계산 중 오류: {str(e)}")
            return 1.0

    def get_processing_statistics(self) -> Dict[str, Any]:
        """처리 통계 정보 반환"""
        return {
            "enabled_features": {
                "agent_preprocessing": self.enable_agent_preprocessing,
                "multimodal_preprocessing": self.enable_multimodal_preprocessing,
                "parallel_processing": self.enable_parallel_processing,
            },
            "configuration": {"max_workers": self.max_workers, "processing_timeout": self.processing_timeout},
            "available_agents": list(self.agents.keys()),
            "available_services": list(self.multimodal_services.keys()),
            "pipeline_steps": len(self.processing_pipeline),
        }
