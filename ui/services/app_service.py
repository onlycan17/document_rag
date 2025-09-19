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
            
            # 우선 제공된 경로가 파일로 존재하면 절대 경로 반환
            if os.path.exists(image_path):
                return os.path.abspath(image_path)
            
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
            
            # 파일을 base64로 인코딩 (폴백)
            # 만약 경로가 없을 경우, 호출자에서 여러 후보 경로를 탐색하도록 설계되어 있으나
            # 안전을 위해 다시 확인하고 가능하면 data URL로 변환합니다.
            if not os.path.exists(image_path):
                return None

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
        print(f"🔍 [AppService.extract_images_from_content] 시작 - 문서 개수: {len(context_documents) if context_documents else 0}")
        images_info: List[Dict[str, Any]] = []

        def _append_image(p: str, md: Dict[str, Any]):
            print(f"🔍 [AppService._append_image] 경로 해결 시도: {p}")
            # 경로 정규화 및 후보 탐색
            resolved = AppService._resolve_image_path(p)
            if not resolved:
                print(f"🔍 [AppService._append_image] ❌ 경로 해결 실패: {p}")
                return
            print(f"🔍 [AppService._append_image] ✅ 경로 해결 성공: {resolved}")
            filename = md.get('filename') or os.path.basename(resolved)
            source = md.get('source') or md.get('file_name') or 'Unknown'
            description = md.get('image_description') or md.get('description') or ''
            image_info = {
                'path': resolved,
                'filename': filename,
                'source': source,
                'description': description
            }
            images_info.append(image_info)
            print(f"🔍 [AppService._append_image] 이미지 정보 추가됨: {image_info}")

        try:
            for i, doc in enumerate(context_documents):
                print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1} 처리 시작")
                # 일부 호출부는 (Document, score) 튜플을 전달할 수 있음
                if isinstance(doc, (list, tuple)) and len(doc) > 0:
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: 튜플/리스트 형태, 첫 번째 요소 추출")
                    doc = doc[0]

                print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1} 타입: {type(doc)}")

                # 문서가 dict 형태로 이미지 정보를 바로 포함할 수도 있음
                if isinstance(doc, dict):
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: dict 형태, 키들: {list(doc.keys())}")
                    # 예: {'image_file': '...'} 형태
                    if 'image_file' in doc and doc.get('image_file'):
                        print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: image_file 발견 - {doc.get('image_file')}")
                        _append_image(doc.get('image_file'), doc)
                        continue
                    if 'image_path' in doc and doc.get('image_path'):
                        print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: image_path 발견 - {doc.get('image_path')}")
                        _append_image(doc.get('image_path'), doc)
                        continue

                metadata = None
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata = doc.metadata
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: hasattr metadata 사용, 키들: {list(metadata.keys()) if metadata else 'None'}")
                elif isinstance(doc, dict) and 'metadata' in doc:
                    metadata = doc.get('metadata')
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: dict metadata 사용, 키들: {list(metadata.keys()) if metadata else 'None'}")

                if not metadata:
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: 메타데이터 없음, 건너뜀")
                    continue
                
                print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: 메타데이터 발견, 키들: {list(metadata.keys())}")

                # 다양한 필드명을 지원: image_paths, image_files, images, intelligent_images, image_path, image_file, relative_path
                # 1) 리스트 형태의 이미지 정보
                if metadata.get('image_paths'):
                    paths = metadata.get('image_paths', [])
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: image_paths 발견 - {len(paths)}개")
                    for p in paths:
                        if p:
                            _append_image(p, metadata)

                elif metadata.get('image_files'):
                    files = metadata.get('image_files', [])
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: image_files 발견 - {len(files)}개")
                    for p in files:
                        if p:
                            _append_image(p, metadata)

                elif metadata.get('images') and isinstance(metadata.get('images'), list):
                    images = metadata.get('images', [])
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: images 발견 - {len(images)}개")
                    for j, item in enumerate(images):
                        print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}, 이미지 {j+1}: {type(item)} - {item if isinstance(item, str) else list(item.keys()) if isinstance(item, dict) else 'unknown'}")
                        if isinstance(item, dict):
                            p = item.get('image_file') or item.get('path') or item.get('relative_path')
                            if p:
                                md = {**metadata, **item}
                                _append_image(p, md)
                        elif isinstance(item, str):
                            _append_image(item, metadata)

                elif metadata.get('intelligent_images') and isinstance(metadata.get('intelligent_images'), list):
                    intelligent_images = metadata.get('intelligent_images', [])
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: intelligent_images 발견 - {len(intelligent_images)}개")
                    for item in intelligent_images:
                        if isinstance(item, dict):
                            p = item.get('image_file') or item.get('image_path') or item.get('path')
                            if p:
                                md = {**metadata, **item}
                                _append_image(p, md)

                # 2) 단일 이미지 필드
                elif metadata.get('image_file'):
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: 단일 image_file 발견 - {metadata.get('image_file')}")
                    _append_image(metadata.get('image_file'), metadata)
                elif metadata.get('image_path'):
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: 단일 image_path 발견 - {metadata.get('image_path')}")
                    _append_image(metadata.get('image_path'), metadata)
                elif metadata.get('relative_path'):
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: 단일 relative_path 발견 - {metadata.get('relative_path')}")
                    # relative_path는 리포지토리 루트 기준 상대 경로일 수 있음
                    _append_image(metadata.get('relative_path'), metadata)
                else:
                    print(f"🔍 [AppService.extract_images_from_content] 문서 {i+1}: 알려진 이미지 필드가 없음")

        except Exception as e:
            print(f"🔍 [AppService.extract_images_from_content] ❌ 전체 오류: {str(e)}")
            logger.error(f"이미지 추출 중 오류: {str(e)}")

        print(f"🔍 [AppService.extract_images_from_content] 완료 - 총 {len(images_info)}개 이미지 반환")
        for i, img in enumerate(images_info):
            print(f"🔍 [AppService.extract_images_from_content] 결과 {i+1}: {img.get('filename')} - {img.get('path')}")
        
        return images_info

    @staticmethod
    def _resolve_image_path(image_ref: str) -> Optional[str]:
        """
        다양한 경우의 이미지 경로를 시도하여 실제 파일 경로를 반환합니다.
        - 절대 경로가 존재하면 즉시 반환
        - static/images/pdf/ 경로는 data/extracted_images/ 경로로 매핑
        - 상대 경로이면 프로젝트 루트와 몇몇 표준 디렉토리에서 탐색
        - 파일명만 주어질 경우 다양한 디렉토리에서 검색
        """
        try:
            if not image_ref:
                return None

            candidate = Path(image_ref)
            # 이미 절대 경로로 존재하면 반환
            if candidate.exists():
                return str(candidate.resolve())

            cwd = Path.cwd()
            
            # static/images/pdf/ 경로를 data/extracted_images/ 경로로 변환
            if image_ref.startswith('static/images/pdf/'):
                # static/images/pdf/filename.ext에서 filename.ext 추출
                filename = os.path.basename(image_ref)
                
                # 파일명에서 문서명과 페이지/이미지 정보 추출
                # 예: 2021년+몽촌토성+북문지+일원+발굴조사+자료집_page002_img031.png
                if '_page' in filename and '_img' in filename:
                    # 기존 형식에서 새 형식으로 변환
                    parts = filename.split('_page')
                    if len(parts) >= 2:
                        doc_name = parts[0]
                        page_img_part = '_page'.join(parts[1:])
                        
                        # page002_img031.png -> p002_i031.png로 변환
                        page_img_part = page_img_part.replace('_page', '_p').replace('_img', '_i')
                        new_filename = f"{doc_name}_p{page_img_part}"
                        
                        # data/extracted_images/doc_name/images/ 경로에서 검색
                        extracted_images_path = cwd / 'data' / 'extracted_images' / doc_name / 'images' / new_filename
                        if extracted_images_path.exists():
                            return str(extracted_images_path.resolve())

            # 후보 디렉토리 목록 - data/extracted_images 우선 추가
            search_dirs = [
                cwd / 'data' / 'extracted_images',
                cwd,
                cwd / 'processed_docs',
                cwd / 'converted_docs',
                cwd / 'data',
                cwd / 'static',
                cwd / 'static' / 'images',
                cwd / 'static' / 'images' / 'pdf',
                cwd / 'static' / 'images' / 'docx',
                cwd / 'prompts',
                cwd / 'docs'
            ]

            # 만약 image_ref이 상대 경로 형태이면 그대로 시도
            for d in search_dirs:
                p = (d / image_ref).resolve()
                if p.exists():
                    return str(p)

            # 파일명만 있는 경우 파일명으로 검색
            name = os.path.basename(image_ref)
            
            # data/extracted_images의 모든 하위 디렉토리에서 검색
            extracted_images_dir = cwd / 'data' / 'extracted_images'
            if extracted_images_dir.exists():
                for doc_dir in extracted_images_dir.iterdir():
                    if doc_dir.is_dir():
                        images_dir = doc_dir / 'images'
                        if images_dir.exists():
                            image_path = images_dir / name
                            if image_path.exists():
                                return str(image_path.resolve())
                                
                            # 다양한 확장자로 시도
                            name_without_ext = os.path.splitext(name)[0]
                            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                                test_path = images_dir / f"{name_without_ext}{ext}"
                                if test_path.exists():
                                    return str(test_path.resolve())
            
            # 기존 검색 디렉토리에서도 시도
            for d in search_dirs:
                p = (d / name).resolve()
                if p.exists():
                    return str(p)

            return None
        except Exception:
            return None
    
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
            # 이미지가 있는 경우 Markdown 형식으로 이미지와 메타데이터를 추가합니다.
            # Streamlit의 chat 컴포넌트는 HTML을 완전히 허용하지 않을 수 있으므로 Markdown 이미지 문법을 사용합니다.
            processed_content += "\n\n---\n\n**📷 관련 이미지:**\n\n"

            for i, image_info in enumerate(images_info[:4], 1):  # 최대 4개 이미지만 표시
                image_data_url = AppService.serve_image(image_info['path'])
                if image_data_url:
                    clean_filename = image_info.get('filename', 'Unknown').replace('+', ' ')
                    clean_source = image_info.get('source', 'Unknown').replace('+', ' ')

                    if len(clean_source) > 35:
                        clean_source = clean_source[:32] + "..."

                    # Markdown 이미지 문법 사용; Streamlit은 Markdown의 이미지 링크를 렌더링합니다.
                    # 크기 제어는 Markdown에서 직접 지원되지 않으므로 필요하면 추후에 st.image로 대체할 수 있습니다.
                    processed_content += f"\n{i}. **{clean_filename}**  \n"
                    processed_content += f"![{clean_filename}]({image_data_url})\n\n"
                    processed_content += f"- 출처: {clean_source}  \n"
                    if image_info.get('description'):
                        processed_content += f"- 설명: {image_info['description']}  \n"
                    processed_content += "\n"
        
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