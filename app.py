import streamlit as st
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# ChromaDB 텔레메트리 비활성화 (가장 먼저 실행)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from config import settings
from src.loaders import DocumentLoader
from src.rag import RAGChain
from src.vectorstore import VectorDatabase
from src.utils.logging_config import setup_logging, get_logger
from src.utils.token_counter import TokenCounter

# 로깅 설정
setup_logging(logging.INFO)
logger = get_logger(__name__)

# 페이지 설정
st.set_page_config(
    page_title=settings.app_title,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': None,
        'Get Help': None,
        'About': settings.app_description
    }
)

# 세션 상태 초기화
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = VectorDatabase()
if 'rag_chain' not in st.session_state:
    # RAG 체인에 벡터 DB 인스턴스를 공유
    st.session_state.rag_chain = RAGChain(vector_db=st.session_state.vector_db)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_loader' not in st.session_state:
    st.session_state.document_loader = DocumentLoader(use_ocr=True)
if 'current_provider' not in st.session_state:
    st.session_state.current_provider = settings.llm_provider
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'token_counter' not in st.session_state:
    st.session_state.token_counter = TokenCounter()

def main():
    st.title(settings.app_title)
    st.markdown(settings.app_description)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 문서 관리 섹션
        st.subheader("📄 문서 관리")
        
        # OCR 옵션
        use_ocr = st.checkbox(
            "PDF OCR 사용", 
            value=True,
            help="스캔된 PDF나 이미지 PDF에서 텍스트를 추출하려면 체크하세요. (Tesseract 필요)"
        )
        
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "문서 업로드 (TXT, MD, PDF)",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True,
            help="대용량 PDF의 경우 OCR 처리 시 시간이 오래 걸릴 수 있습니다."
        )
        
        if uploaded_files:
            if st.button("문서 처리 및 저장"):
                for uploaded_file in uploaded_files:
                    # 파일별 진행 상황 표시
                    file_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 임시 파일로 저장
                    temp_path = f"./data/documents/{uploaded_file.name}"
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 파일 크기 확인
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    file_placeholder.info(f"📄 처리 중: **{uploaded_file.name}** ({file_size_mb:.1f}MB)")
                    
                    # 진행 상황 콜백
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    # 문서 로드 및 벡터 DB에 추가
                    try:
                        logger.info(f"문서 처리 시작: {uploaded_file.name} ({file_size_mb:.1f}MB)")
                        
                        # OCR 설정 업데이트
                        if st.session_state.document_loader.use_ocr != use_ocr:
                            st.session_state.document_loader = DocumentLoader(use_ocr=use_ocr)
                        
                        # 문서 로드 (진행 상황 콜백 포함)
                        documents = st.session_state.document_loader.load_document(temp_path, update_progress)
                        
                        # 벡터 DB에 추가
                        update_progress(0.95, "벡터 데이터베이스에 저장 중...")
                        st.session_state.vector_db.add_documents(documents)
                        
                        # 완료
                        update_progress(1.0, "완료!")
                        
                        # 메타데이터 확인
                        extraction_method = documents[0].metadata.get('extraction_method', 'standard') if documents else 'unknown'
                        file_placeholder.success(f"✅ {uploaded_file.name} 처리 완료 ({len(documents)} 청크) - 방법: {extraction_method}")
                        logger.info(f"문서 처리 완료: {uploaded_file.name} - {len(documents)} 청크, 방법: {extraction_method}")
                        
                        # 진행 바와 상태 텍스트 제거
                        progress_bar.empty()
                        status_text.empty()
                        
                    except Exception as e:
                        logger.error(f"문서 처리 실패: {uploaded_file.name} - {str(e)}")
                        file_placeholder.error(f"❌ {uploaded_file.name} 처리 실패: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
                
                # 처리 완료 후 새로고침
                import time
                time.sleep(1)  # 사용자가 결과를 확인할 수 있도록 잠시 대기
                st.rerun()
        
        # 기존 문서 로드
        if st.button("domain.md 파일 로드"):
            domain_path = "./domain.md"
            if os.path.exists(domain_path):
                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                try:
                    documents = st.session_state.document_loader.load_document(domain_path, update_progress)
                    
                    update_progress(0.95, "벡터 데이터베이스에 저장 중...")
                    st.session_state.vector_db.add_documents(documents)
                    
                    update_progress(1.0, "완료!")
                    st.success(f"✅ domain.md 로드 완료 ({len(documents)} 청크)")
                    
                    # 진행 표시 제거
                    progress_bar.empty()
                    status_text.empty()
                except Exception as e:
                    st.error(f"❌ 파일 로드 실패: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
            else:
                st.error("domain.md 파일을 찾을 수 없습니다.")
        
        # 벡터 DB 상태
        st.divider()
        doc_count = st.session_state.vector_db.get_document_count()
        st.info(f"💾 저장된 문서 청크: {doc_count}개")
        
        # 로그 뷰어 (확장 가능)
        with st.expander("📋 처리 로그 보기"):
            log_file = f"./logs/rag_app_{datetime.now().strftime('%Y%m%d')}.log"
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    # 최근 50줄만 표시
                    lines = f.readlines()
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    st.text(''.join(recent_lines))
            else:
                st.text("로그 파일이 없습니다.")
        
        # 벡터 DB 초기화
        col1, col2 = st.columns(2)
        with col1:
            if st.button("벡터 DB 초기화", type="secondary", key="clear_db_btn"):
                st.session_state.show_clear_confirm = True
        
        if 'show_clear_confirm' in st.session_state and st.session_state.show_clear_confirm:
            with col2:
                if st.button("⚠️ 확인", type="primary", key="confirm_clear"):
                    st.session_state.vector_db.clear_database()
                    # RAG 체인도 재초기화 (벡터 DB 인스턴스 공유)
                    st.session_state.rag_chain = RAGChain(
                        provider=st.session_state.current_provider, 
                        model=st.session_state.current_model,
                        vector_db=st.session_state.vector_db
                    )
                    st.success("벡터 데이터베이스가 초기화되었습니다.")
                    st.session_state.show_clear_confirm = False
                    st.rerun()
        
        # LLM 설정
        st.divider()
        st.subheader("🤖 LLM 모델 설정")
        
        # 사용 가능한 모델 가져오기
        available_models = st.session_state.rag_chain.get_available_models()
        
        # LLM 제공자 선택
        provider_names = {
            "openai": "OpenAI",
            "google": "Google Gemini",
            "anthropic": "Anthropic Claude",
            "local": "로컬 LLM"
        }
        
        selected_provider = st.selectbox(
            "LLM 제공자 선택",
            options=list(provider_names.keys()),
            format_func=lambda x: provider_names[x],
            index=list(provider_names.keys()).index(st.session_state.current_provider)
        )
        
        # 선택된 제공자의 모델 목록
        if selected_provider in available_models:
            model_options = available_models[selected_provider]
            
            # 모델 선택
            selected_model_info = st.selectbox(
                "모델 선택",
                options=model_options,
                format_func=lambda x: f"{x['name']} - {x['description']}",
                index=0
            )
            
            selected_model = selected_model_info['model'] if selected_model_info else None
            
            # API 키 확인
            api_key_status = "✅ API 키 설정됨"
            if selected_provider == "openai" and not settings.openai_api_key:
                api_key_status = "❌ OpenAI API 키가 필요합니다"
            elif selected_provider == "google" and not settings.google_api_key:
                api_key_status = "❌ Google API 키가 필요합니다"
            elif selected_provider == "anthropic" and not settings.anthropic_api_key:
                api_key_status = "❌ Anthropic API 키가 필요합니다"
            elif selected_provider == "local":
                api_key_status = "✅ 로컬 모델 (API 키 불필요)"
            
            st.caption(api_key_status)
            
            # 모델 변경 버튼
            if st.button("모델 적용", type="primary"):
                try:
                    st.session_state.rag_chain.update_llm(selected_provider, selected_model)
                    st.session_state.current_provider = selected_provider
                    st.session_state.current_model = selected_model
                    st.success(f"✅ {provider_names[selected_provider]} - {selected_model_info['name']} 모델로 변경되었습니다!")
                except Exception as e:
                    st.error(f"❌ 모델 변경 실패: {str(e)}")
        
        # 고급 설정
        st.divider()
        st.subheader("🔧 고급 설정")
        
        k_documents = st.slider(
            "검색할 문서 개수",
            min_value=1,
            max_value=10,
            value=settings.k_documents,
            help="더 많은 문서를 검색하면 더 포괄적인 답변을 얻을 수 있습니다."
        )
        settings.k_documents = k_documents
        
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=settings.temperature,
            step=0.1,
            help="낮을수록 일관성 있고, 높을수록 창의적인 답변"
        )
        settings.temperature = temperature
        
        # 디버그 모드
        debug_mode = st.checkbox("🐛 디버그 모드", help="검색 결과와 점수를 표시합니다.")
        st.session_state.debug_mode = debug_mode
    
    # 메인 챗 인터페이스
    st.header("💬 챗봇")
    
    # 현재 사용 중인 모델 및 컨텍스트 사용량 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        current_model_display = f"현재 모델: **{provider_names.get(st.session_state.current_provider, st.session_state.current_provider)}**"
        if st.session_state.current_model:
            current_model_display += f" - {st.session_state.current_model}"
        st.caption(current_model_display)
    
    with col2:
        # 컨텍스트 사용량 계산 및 표시
        if st.session_state.messages:
            # 현재 대화 내역에서 텍스트 추출
            messages_for_count = []
            for msg in st.session_state.messages:
                messages_for_count.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 마지막 검색된 문서의 토큰 수 (대략적)
            documents_text = " " * (st.session_state.rag_chain.get_last_context_tokens() * 4)
            
            # 컨텍스트 사용량 계산
            usage_info = st.session_state.token_counter.calculate_context_usage(
                messages_for_count,
                documents_text,
                st.session_state.current_provider,
                st.session_state.current_model or "gpt-3.5-turbo"
            )
            
            # 사용량 표시
            usage_display = st.session_state.token_counter.format_token_display(usage_info)
            st.caption(f"컨텍스트: {usage_display}")
            
            # 상세 정보 툴팁
            with st.expander("📊 토큰 사용량 상세"):
                st.markdown(st.session_state.token_counter.get_usage_breakdown(usage_info))
        else:
            st.caption("컨텍스트: 🟢 0 토큰 사용 중")
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 출처 정보 표시
            if "sources" in message and message["sources"]:
                with st.expander("📌 참고 문서"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**문서 {i+1}**: {source['file_name']}")
                        st.markdown(f"- 관련도: {1 - source['relevance_score']:.2%}")
                        st.markdown(f"- 내용 미리보기: {source['content_preview']}")
                        st.divider()
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 봇 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                response = st.session_state.rag_chain.query(prompt)
                
                # 실제 검색된 문서의 토큰 수 가져오기
                context_tokens = st.session_state.rag_chain.get_last_context_tokens()
                
                # 디버그 모드일 때 검색 결과 표시
                if st.session_state.debug_mode:
                    with st.expander("🔍 디버그 정보"):
                        st.write(f"**검색 쿼리**: {prompt}")
                        st.write(f"**벡터 DB 상태**:")
                        st.write(f"- 총 문서 청크 수: {st.session_state.vector_db.get_document_count()}")
                        st.write(f"- 벡터 DB 타입: {settings.vector_db_type}")
                        st.write(f"- 검색 임계값: FAISS < 2.0, ChromaDB > 0.2")
                        st.write(f"**검색 설정**:")
                        st.write(f"- 검색할 문서 수 (k): {settings.k_documents}")
                        st.write(f"**검색 결과**:")
                        st.write(f"- 검색된 문서 수: {len(response.get('sources', []))}")
                        st.write(f"- 응답 상태: {response.get('status', 'unknown')}")
                        
                        if response.get('sources'):
                            st.write("**문서별 상세 정보**:")
                            for i, source in enumerate(response['sources']):
                                st.write(f"- **문서 {i+1}**: {source['file_name']}")
                                st.write(f"  - 관련도 점수: {source['relevance_score']:.4f}")
                                st.write(f"  - 소스 경로: {source['source_path']}")
                                st.write(f"  - 내용 미리보기: {source['content_preview'][:100]}...")
                                st.write("---")
                        else:
                            st.write("❌ 검색된 문서가 없습니다.")
                            
                            # 벡터 DB에 저장된 문서가 있다면 원시 검색 시도
                            if st.session_state.vector_db.get_document_count() > 0:
                                try:
                                    if hasattr(st.session_state.vector_db.vector_store, 'similarity_search_with_score'):
                                        raw_results = st.session_state.vector_db.vector_store.similarity_search_with_score(prompt, k=3)
                                        st.write(f"**원시 검색 결과** (임계값 무시):")
                                        for i, (doc, score) in enumerate(raw_results):
                                            st.write(f"- 문서 {i+1}: 점수 {score:.4f}, 내용: {doc.page_content[:100]}...")
                                except Exception as e:
                                    st.write(f"원시 검색 실패: {str(e)}")
                
                if response["status"] == "success":
                    st.markdown(response["answer"])
                    
                    # 응답 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                    
                    # 출처 정보 표시
                    if response["sources"]:
                        with st.expander("📌 참고 문서"):
                            for i, source in enumerate(response["sources"]):
                                st.markdown(f"**문서 {i+1}**: {source['file_name']}")
                                st.markdown(f"- 관련도: {1 - source['relevance_score']:.2%}")
                                st.markdown(f"- 내용 미리보기: {source['content_preview']}")
                                st.divider()
                else:
                    st.error(response["answer"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })
            
            # 컨텍스트 사용량 업데이트를 위해 리런
            st.rerun()
    
    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()