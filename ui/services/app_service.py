"""
애플리케이션 서비스 레이어
"""

import os
import time
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging

# pdf_metadata_extractor와 search_quality_tester는 추후 구현 예정

logger = logging.getLogger(__name__)


class AppService:
    """애플리케이션의 비즈니스 서비스를 제공하는 클래스"""
    
    @staticmethod
    def serve_image(image_path: str) -> Optional[str]:
        """
        이미지 파일을 base64 데이터 URL로 변환합니다.
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Optional[str]: base64 데이터 URL 또는 None
        """
        try:
            import base64
            from pathlib import Path
            
            if not os.path.exists(image_path):
                return None
            
            # 파일 확장자 확인
            ext = Path(image_path).suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                return None
            
            # MIME 타입 결정
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # 파일을 base64로 인코딩
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            return f"data:{mime_type};base64,{image_data}"
            
        except Exception as e:
            logger.error(f"이미지 변환 실패: {image_path} - {str(e)}")
            return None
    
    @staticmethod
    def extract_images_from_content(content: str, context_documents: List[Any]) -> List[Dict[str, Any]]:
        """
        컨텐츠와 컨텍스트 문서에서 이미지 정보를 추출합니다.
        
        Args:
            content: 응답 컨텐츠
            context_documents: 컨텍스트 문서 리스트
            
        Returns:
            List[Dict[str, Any]]: 이미지 정보 리스트
        """
        images_info = []
        
        try:
            for doc in context_documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata = doc.metadata
                    
                    # 이미지 경로 정보가 있는지 확인
                    if 'image_paths' in metadata and metadata['image_paths']:
                        for image_path in metadata['image_paths']:
                            if os.path.exists(image_path):
                                # 파일명과 소스 정보 추출
                                filename = os.path.basename(image_path)
                                source = metadata.get('source', 'Unknown')
                                description = metadata.get('image_description', '')
                                
                                images_info.append({
                                    'path': image_path,
                                    'filename': filename,
                                    'source': source,
                                    'description': description
                                })
                    
                    # 단일 이미지 경로가 있는지 확인
                    elif 'image_path' in metadata and metadata['image_path']:
                        image_path = metadata['image_path']
                        if os.path.exists(image_path):
                            filename = os.path.basename(image_path)
                            source = metadata.get('source', 'Unknown')
                            description = metadata.get('image_description', '')
                            
                            images_info.append({
                                'path': image_path,
                                'filename': filename,
                                'source': source,
                                'description': description
                            })
            
        except Exception as e:
            logger.error(f"이미지 추출 중 오류: {str(e)}")
        
        return images_info
    
    @staticmethod
    def display_images_in_response(content: str, context_documents: List[Any]) -> str:
        """
        응답에 이미지를 표시하는 HTML을 생성합니다.
        
        Args:
            content: 원본 컨텐츠
            context_documents: 컨텍스트 문서 리스트
            
        Returns:
            str: 이미지가 포함된 HTML 컨텐츠
        """
        # 기본 컨텐츠
        processed_content = content
        
        # 이미지 추출
        images_info = AppService.extract_images_from_content(content, context_documents)
        
        if images_info:
            # 이미지가 있는 경우 HTML 추가
            processed_content += "\n\n---\n\n**📷 관련 이미지:**\n\n"
            
            for i, image_info in enumerate(images_info[:4], 1):  # 최대 4개 이미지만 표시
                image_data_url = AppService.serve_image(image_info['path'])
                if image_data_url:
                    clean_filename = image_info.get('filename', 'Unknown').replace('+', ' ')
                    clean_source = image_info.get('source', 'Unknown').replace('+', ' ')
                    
                    if len(clean_source) > 35:
                        clean_source = clean_source[:32] + "..."
                    
                    # 간단한 이미지 표시
                    processed_content += f"""
                    <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                        <img src="{image_data_url}" style="max-width: 300px; max-height: 200px; border-radius: 3px;" />
                        <br>
                        <small><strong>파일:</strong> {clean_filename}</small><br>
                        <small><strong>출처:</strong> {clean_source}</small>
                    """
                    
                    if image_info.get('description'):
                        processed_content += f"<br><small><strong>설명:</strong> {image_info['description']}</small>"
                    
                    processed_content += "</div>\n\n"
        
        return processed_content
    
    @staticmethod
    def format_chat_message_with_metadata(message: Dict[str, Any]) -> str:
        """
        채팅 메시지에 메타데이터를 포함하여 포맷합니다.
        
        Args:
            message: 채팅 메시지 딕셔너리
            
        Returns:
            str: 포맷된 메시지
        """
        content = message.get('content', '')
        
        # 소스 정보가 있는 경우 추가
        if 'sources' in message and message['sources']:
            content += "\n\n**📌 참고 문서:**\n"
            for i, source in enumerate(message['sources'][:3], 1):  # 최대 3개만 표시
                if isinstance(source, dict):
                    file_name = source.get('file_name', 'Unknown')
                    relevance_score = source.get('relevance_score', 0)
                    content_preview = source.get('content_preview', '')
                else:
                    metadata = getattr(source, 'metadata', {})
                    file_name = metadata.get('file_name', 'Unknown')
                    relevance_score = 0
                    content_preview = source.page_content[:100] if hasattr(source, 'page_content') else ''
                
                content += f"\n{i}. **{file_name}**"
                if relevance_score > 0:
                    content += f" (관련도: {1 - relevance_score:.1%})"
                if content_preview:
                    content += f"\n   _{content_preview}..._"
        
        return content
    
    @staticmethod
    def extract_source_information(response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        응답 데이터에서 소스 정보를 추출합니다.
        
        Args:
            response_data: RAG 응답 데이터
            
        Returns:
            List[Dict[str, Any]]: 소스 정보 리스트
        """
        sources = []
        
        try:
            context_documents = response_data.get('context_documents', [])
            
            for doc in context_documents:
                if hasattr(doc, 'metadata'):
                    metadata = doc.metadata
                    source_info = {
                        'file_name': metadata.get('source', 'Unknown'),
                        'page': metadata.get('page', 'N/A'),
                        'relevance_score': metadata.get('relevance_score', 0),
                        'content_preview': doc.page_content[:200] if hasattr(doc, 'page_content') else ''
                    }
                    sources.append(source_info)
                else:
                    # 메타데이터가 없는 경우 기본 정보
                    source_info = {
                        'file_name': 'Unknown',
                        'page': 'N/A',
                        'relevance_score': 0,
                        'content_preview': str(doc)[:200] if doc else ''
                    }
                    sources.append(source_info)
        
        except Exception as e:
            logger.error(f"소스 정보 추출 중 오류: {str(e)}")
        
        return sources
    
    @staticmethod
    def generate_chat_export_data() -> str:
        """
        채팅 기록을 내보내기용 텍스트로 변환합니다.
        
        Returns:
            str: 내보내기용 텍스트
        """
        if not st.session_state.get('messages'):
            return ""
        
        # 채팅 기록을 텍스트로 변환
        chat_text = f"# 채팅 기록 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for i, message in enumerate(st.session_state.messages, 1):
            role = "사용자" if message["role"] == "user" else "어시스턴트"
            content = message["content"]
            
            chat_text += f"## {i}. {role}\n\n{content}\n\n"
            
            # 성능 정보 추가 (어시스턴트 메시지인 경우)
            if message["role"] == "assistant" and "processing_time" in message:
                processing_time = message["processing_time"]
                context_count = len(message.get("context_documents", []))
                chat_text += f"*처리 시간: {processing_time:.2f}초, 참조 문서: {context_count}개*\n\n"
            
            chat_text += "---\n\n"
        
        return chat_text
    
    @staticmethod
    def process_streaming_response(
        stream_generator,
        response_placeholder: Any,
        status_placeholder: Any,
        is_local_model: bool = False
    ) -> Dict[str, Any]:
        """
        스트리밍 응답을 처리합니다.
        
        Args:
            stream_generator: 스트리밍 제너레이터
            response_placeholder: 응답 표시 위젯
            status_placeholder: 상태 표시 위젯
            is_local_model: 로컬 모델 여부
            
        Returns:
            Dict[str, Any]: 처리된 응답 데이터
        """
        full_response = ""
        response_sources = []
        response_status = "unknown"
        response_search_info = {}
        
        try:
            for chunk in stream_generator:
                chunk_type = chunk.get("type", "unknown")
                chunk_content = chunk.get("content", "")
                
                if chunk_type == "status":
                    # 상태 메시지 표시
                    if is_local_model:
                        # 로컬 모델용 상태 메시지 커스터마이징
                        if "검색" in chunk_content:
                            status_placeholder.info(chunk_content)
                        elif "생성" in chunk_content:
                            status_placeholder.warning("⏳ 로컬 모델이 답변을 생성 중입니다. 시간이 걸릴 수 있습니다...")
                    else:
                        status_placeholder.info(chunk_content)
                
                elif chunk_type == "response":
                    # 응답 텍스트 누적 및 표시
                    full_response += chunk_content
                    response_placeholder.markdown(full_response)
                
                elif chunk_type == "sources":
                    # 소스 정보 저장
                    response_sources = chunk_content
                
                elif chunk_type == "search_info":
                    # 검색 정보 저장
                    response_search_info = chunk_content
                
                elif chunk_type == "final":
                    # 최종 응답 처리
                    response_status = "success"
                    final_data = chunk_content
                    if isinstance(final_data, dict):
                        full_response = final_data.get('answer', full_response)
                        response_sources = final_data.get('context_documents', response_sources)
                    break
                
                elif chunk_type == "error":
                    # 오류 처리
                    response_status = "error"
                    status_placeholder.error(f"스트리밍 중 오류: {chunk_content}")
                    break
            
            # 상태 메시지 정리
            status_placeholder.empty()
            
            return {
                'status': response_status,
                'answer': full_response,
                'context_documents': response_sources,
                'search_info': response_search_info
            }
        
        except Exception as e:
            logger.error(f"스트리밍 응답 처리 중 오류: {str(e)}")
            status_placeholder.error(f"스트리밍 처리 중 오류: {str(e)}")
            return {
                'status': 'error',
                'answer': '',
                'context_documents': [],
                'error': str(e)
            }
    
    @staticmethod
    def validate_file_upload(uploaded_file: Any) -> Dict[str, Any]:
        """
        업로드된 파일을 검증합니다.
        
        Args:
            uploaded_file: 업로드된 파일 객체
            
        Returns:
            Dict[str, Any]: 검증 결과
        """
        try:
            # 파일 크기 확인 (100MB 제한)
            max_size = 100 * 1024 * 1024  # 100MB
            if uploaded_file.size > max_size:
                return {
                    'valid': False,
                    'error': f"파일 크기가 너무 큽니다. (최대 100MB, 현재: {uploaded_file.size / (1024*1024):.1f}MB)"
                }
            
            # 파일 확장자 확인
            allowed_extensions = ['.txt', '.md', '.pdf', '.docx']
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in allowed_extensions:
                return {
                    'valid': False,
                    'error': f"지원하지 않는 파일 형식입니다. ({file_ext})"
                }
            
            # 파일명 유효성 확인
            if not uploaded_file.name or len(uploaded_file.name.strip()) == 0:
                return {
                    'valid': False,
                    'error': "유효하지 않은 파일명입니다."
                }
            
            return {
                'valid': True,
                'error': None,
                'file_info': {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'extension': file_ext
                }
            }
        
        except Exception as e:
            return {
                'valid': False,
                'error': f"파일 검증 중 오류: {str(e)}"
            }
    
    @staticmethod
    def get_performance_metrics(
        processing_time: float,
        metadata: Dict[str, Any],
        context_documents: List[Any]
    ) -> Dict[str, Any]:
        """
        성능 메트릭을 계산합니다.
        
        Args:
            processing_time: 처리 시간
            metadata: 메타데이터
            context_documents: 컨텍스트 문서
            
        Returns:
            Dict[str, Any]: 성능 메트릭
        """
        return {
            'processing_time': processing_time,
            'document_count': len(context_documents),
            'total_tokens': metadata.get('total_tokens', 0),
            'prompt_tokens': metadata.get('prompt_tokens', 0),
            'completion_tokens': metadata.get('completion_tokens', 0),
            'model': metadata.get('model', 'Unknown'),
            'provider': metadata.get('provider', 'Unknown')
        }
    
    @staticmethod
    def create_debug_info(
        query: str,
        response_data: Dict[str, Any],
        processing_time: float
    ) -> Dict[str, Any]:
        """
        디버그 정보를 생성합니다.
        
        Args:
            query: 사용자 쿼리
            response_data: 응답 데이터
            processing_time: 처리 시간
            
        Returns:
            Dict[str, Any]: 디버그 정보
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'processing_time': processing_time,
            'response_length': len(response_data.get('answer', '')),
            'context_document_count': len(response_data.get('context_documents', [])),
            'metadata': response_data.get('metadata', {}),
            'search_info': response_data.get('search_info', {}),
            'model_info': {
                'provider': st.session_state.get('current_provider', 'Unknown'),
                'model': st.session_state.get('current_model', 'Unknown')
            }
        }