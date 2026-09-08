"""
메인 애플리케이션 컨트롤러
"""

import streamlit as st
from typing import Dict, Any

from src.rag.rag_chain import RAGChain
from src.vectorstore.vector_db import VectorDatabase


class MainController:
    """메인 애플리케이션의 비즈니스 로직을 처리하는 컨트롤러"""

    def __init__(self):
        """컨트롤러 초기화"""
        self.vector_db = None
        self.rag_chain = None

    def safe_get_vector_db(self) -> VectorDatabase:
        """
        안전하게 벡터 데이터베이스 인스턴스를 가져옵니다.

        Returns:
            VectorDatabase: 벡터 데이터베이스 인스턴스
        """
        if "vector_db" not in st.session_state:
            st.session_state.vector_db = VectorDatabase()
        return st.session_state.vector_db

    def safe_get_rag_chain(self) -> RAGChain:
        """
        안전하게 RAG 체인 인스턴스를 가져옵니다.

        Returns:
            RAGChain: RAG 체인 인스턴스
        """
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = RAGChain(vector_db=self.safe_get_vector_db())
        return st.session_state.rag_chain

    def bootstrap_models_with_file_lock(self, provider: str, model: str) -> bool:
        """
        파일 락을 사용하여 모델을 부트스트랩합니다.

        Args:
            provider: 모델 공급자
            model: 모델 이름

        Returns:
            bool: 성공 여부
        """
        try:
            rag_chain = self.safe_get_rag_chain()

            # 모델 변경
            success = rag_chain.update_llm(provider, model)

            if success:
                # 세션 상태 업데이트
                st.session_state.current_provider = provider
                st.session_state.current_model = model
                return True

            return False

        except Exception as e:
            st.error(f"모델 부트스트랩 실패: {str(e)}")
            return False

    def get_preprocessing_settings_from_sidebar(self, sidebar_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        사이드바 설정에서 전처리 설정을 추출합니다.

        Args:
            sidebar_config: 사이드바 설정 딕셔너리

        Returns:
            Dict[str, Any]: 전처리 설정 딕셔너리
        """
        return {
            "use_ocr": sidebar_config.get("use_ocr", True),
            "use_agent_mode": sidebar_config.get("use_agent_mode", False),
            "enable_postprocessing": sidebar_config.get("enable_postprocessing", False),
            "use_intelligent_extraction": sidebar_config.get("use_intelligent_extraction", True),  # 기본값: True
            "selected_preprocessing_model": sidebar_config.get("selected_preprocessing_model", "local"),
        }

    def initialize_session_state(self) -> None:
        """세션 상태를 초기화합니다."""

        # 기본 세션 상태 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "document_loader" not in st.session_state:
            from config import settings as _settings
            from src.loaders.document_loader import DocumentLoader

            st.session_state.document_loader = DocumentLoader(
                use_ocr=True,
                use_agent_preprocessing=False,
                enable_postprocessing=False,
                use_intelligent_image_extraction=True,  # 기본값: True (권장)
                preprocessing_model=_settings.preprocessing_model,
                enable_multimodal_preprocessing=False,
            )

        if "current_provider" not in st.session_state:
            from config import settings

            st.session_state.current_provider = settings.llm_provider

        if "current_model" not in st.session_state:
            st.session_state.current_model = None

        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False

        if "token_counter" not in st.session_state:
            from src.utils.token_counter import TokenCounter

            st.session_state.token_counter = TokenCounter()

        if "preprocessing_model" not in st.session_state:
            st.session_state.preprocessing_model = "local"

        if "enable_multimodal_preprocessing" not in st.session_state:
            from config import settings

            st.session_state.enable_multimodal_preprocessing = settings.enable_multimodal_preprocessing

        if "intelligent_extraction" not in st.session_state:
            st.session_state.intelligent_extraction = False

        if "sidebar_config" not in st.session_state:
            st.session_state.sidebar_config = {}

    def handle_provider_model_change(self, new_provider: str, new_model: str) -> bool:
        """
        공급자/모델 변경을 처리합니다.

        Args:
            new_provider: 새로운 공급자
            new_model: 새로운 모델

        Returns:
            bool: 변경 성공 여부
        """
        current_provider = st.session_state.get("current_provider")
        current_model = st.session_state.get("current_model")

        # 변경이 필요한지 확인
        if new_provider != current_provider or new_model != current_model:
            return self.bootstrap_models_with_file_lock(new_provider, new_model)

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        현재 모델 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 모델 정보
        """
        rag_chain = self.safe_get_rag_chain()

        return {
            "provider": st.session_state.get("current_provider", "Unknown"),
            "model": st.session_state.get("current_model", "Unknown"),
            "available_models": rag_chain.get_available_models(),
        }

    def get_vector_db_status(self) -> Dict[str, Any]:
        """
        벡터 DB 상태 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 벡터 DB 상태 정보
        """
        vector_db = self.safe_get_vector_db()

        return {"document_count": vector_db.get_document_count(), "is_initialized": vector_db is not None}

    def clear_chat_history(self) -> None:
        """채팅 기록을 삭제합니다."""
        st.session_state.messages = []

    def process_chat_query(self, query: str) -> Dict[str, Any]:
        """
        채팅 쿼리를 처리합니다.

        Args:
            query: 사용자 쿼리

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            rag_chain = self.safe_get_rag_chain()
            response_data = rag_chain.invoke(query)

            return {"success": True, "response": response_data, "error": None}

        except Exception as e:
            return {"success": False, "response": None, "error": str(e)}

    def process_streaming_query(self, query: str):
        """
        스트리밍 쿼리를 처리합니다.

        Args:
            query: 사용자 쿼리

        Yields:
            스트리밍 응답 청크
        """
        try:
            rag_chain = self.safe_get_rag_chain()
            for chunk in rag_chain.stream_query(query):
                yield chunk

        except Exception as e:
            yield {"type": "error", "content": str(e)}

    def update_preprocessing_settings(self, settings: Dict[str, Any]) -> None:
        """
        전처리 설정을 업데이트합니다.

        Args:
            settings: 새로운 전처리 설정
        """
        from src.loaders.document_loader import DocumentLoader

        st.session_state.document_loader = DocumentLoader(
            use_ocr=settings.get("use_ocr", True),
            use_agent_preprocessing=settings.get("use_agent_mode", False),
            enable_postprocessing=settings.get("enable_postprocessing", False),
            use_intelligent_image_extraction=settings.get("use_intelligent_extraction", False),
            preprocessing_model=settings.get("selected_preprocessing_model", "openrouter"),
            enable_multimodal_preprocessing=st.session_state.enable_multimodal_preprocessing,
        )

    def validate_configuration(self) -> Dict[str, Any]:
        """
        현재 설정을 검증합니다.

        Returns:
            Dict[str, Any]: 검증 결과
        """
        issues = []
        warnings = []

        # RAG 체인 검증
        try:
            rag_chain = self.safe_get_rag_chain()
            if not rag_chain:
                issues.append("RAG 체인이 초기화되지 않았습니다.")
        except Exception as e:
            issues.append(f"RAG 체인 검증 실패: {str(e)}")

        # 벡터 DB 검증
        try:
            vector_db = self.safe_get_vector_db()
            if not vector_db:
                issues.append("벡터 데이터베이스가 초기화되지 않았습니다.")
            elif vector_db.get_document_count() == 0:
                warnings.append("벡터 데이터베이스에 문서가 없습니다.")
        except Exception as e:
            issues.append(f"벡터 DB 검증 실패: {str(e)}")

        # 모델 설정 검증
        if not st.session_state.get("current_provider"):
            issues.append("LLM 공급자가 설정되지 않았습니다.")

        if not st.session_state.get("current_model"):
            warnings.append("LLM 모델이 설정되지 않았습니다.")

        return {"is_valid": len(issues) == 0, "issues": issues, "warnings": warnings}
