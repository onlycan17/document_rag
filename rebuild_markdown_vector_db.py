#!/usr/bin/env python3
"""
마크다운 파일 전용 벡터 데이터베이스 구축 스크립트
- data/documents 폴더의 마크다운 파일만 처리
- 한국어 최적화 텍스트 처리
- 상세한 진행 상황 로깅
- 안전한 오류 처리 및 복구
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.vectorstore import VectorDatabase
from src.embeddings import EmbeddingModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import shutil
import os
import time
import logging
from config import settings
from typing import List, Tuple
import traceback

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarkdownDocumentProcessor:
    """마크다운 문서 전용 처리기"""
    
    def __init__(self):
        """마크다운 처리에 최적화된 텍스트 분할기 초기화"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            # 마크다운 구조를 고려한 분리자 순서
            separators=[
                "\n\n\n",    # 여러 줄바꿈 (섹션 구분)
                "\n\n",      # 문단 분리
                "\n#",       # 제목 구분
                "\n##",      # 하위 제목 구분
                "\n###",     # 세부 제목 구분
                "\n- ",      # 목록 항목
                "\n* ",      # 목록 항목 (별표)
                "\n",        # 일반 줄바꿈
                "。",        # 한국어 마침표
                ".",         # 영어 마침표
                "!",         # 느낌표
                "?",         # 물음표
                ";",         # 세미콜론
                ",",         # 쉼표
                " ",         # 공백
                ""           # 문자 단위
            ]
        )
        logger.info(f"📝 마크다운 전용 텍스트 분할기 초기화 완료")
        logger.info(f"   청크 크기: {settings.chunk_size}자")
        logger.info(f"   겹침 크기: {settings.chunk_overlap}자")
    
    def load_markdown_file(self, file_path: str) -> Tuple[List[Document], str]:
        """
        마크다운 파일을 안전하게 로딩
        Returns: (documents, status_message)
        """
        try:
            start_time = time.time()
            file_name = Path(file_path).name
            
            # 다양한 인코딩으로 시도
            encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"   인코딩 {encoding} 시도 실패: {str(e)}")
                    continue
            
            if content is None:
                return [], "마크다운 파일 로딩 실패: 지원되는 인코딩 없음"
            
            load_time = time.time() - start_time
            logger.info(f"   📖 마크다운 로딩 완료: {used_encoding} 인코딩 ({load_time:.1f}초)")
            
            # 마크다운 특화 텍스트 정제
            cleaned_content = self._clean_markdown_text(content)
            
            # Document 객체 생성
            document = Document(
                page_content=cleaned_content,
                metadata={
                    'source': str(file_path),
                    'file_name': file_name,
                    'file_type': '.md',
                    'encoding': used_encoding,
                    'original_size': len(content),
                    'cleaned_size': len(cleaned_content)
                }
            )
            
            status = f"마크다운 로딩 성공: {used_encoding} 인코딩"
            return [document], status
            
        except Exception as e:
            error_msg = f"마크다운 파일 로딩 실패: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            logger.debug(f"   📋 상세 오류:\n{traceback.format_exc()}")
            return [], error_msg
    
    def _clean_markdown_text(self, text: str) -> str:
        """마크다운 특화 텍스트 정제"""
        if not text:
            return ""
        
        import re
        
        # 기본 정제
        text = text.strip()
        
        # 마크다운 메타데이터 제거 (YAML front matter)
        text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)
        
        # 불필요한 마크다운 구문 정리 (내용은 보존하되 구문만 정리)
        # 이미지 링크는 텍스트만 추출
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        
        # 링크는 텍스트만 추출
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # 코드 블록 표시 제거 (내용은 유지)
        text = re.sub(r'```[a-zA-Z]*\n', '', text)
        text = text.replace('```', '')
        
        # 인라인 코드 표시 제거
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # 강조 표시 제거 (내용은 유지)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 굵은 글씨
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # 이탤릭
        text = re.sub(r'__([^_]+)__', r'\1', text)      # 굵은 글씨
        text = re.sub(r'_([^_]+)_', r'\1', text)        # 이탤릭
        
        # 제목 표시 정리 (# 기호 제거하되 제목은 유지)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # 목록 표시 정리
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # 일반적인 텍스트 정제
        text = self._general_text_cleaning(text)
        
        return text
    
    def _general_text_cleaning(self, text: str) -> str:
        """일반적인 텍스트 정제"""
        import re
        
        # 특수 문자 정리
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # BOM
        text = text.replace('\xa0', ' ')   # Non-breaking space
        text = text.replace('\u3000', ' ') # Ideographic space
        
        # 연속된 공백 제거
        text = re.sub(r' +', ' ', text)
        
        # 연속된 줄바꿈 정리 (3개 이상을 2개로)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 한국어 문장 부호 정규화
        text = text.replace('．', '.')
        text = text.replace('，', ',')
        text = text.replace('；', ';')
        text = text.replace('：', ':')
        
        return text.strip()
    
    def chunk_documents(self, documents: List[Document]) -> Tuple[List[Document], str]:
        """
        마크다운 문서를 의미 단위로 청크 분할
        Returns: (chunks, status_message)
        """
        if not documents:
            return [], "분할할 문서가 없음"
        
        try:
            start_time = time.time()
            chunks = self.text_splitter.split_documents(documents)
            chunk_time = time.time() - start_time
            
            # 청크 품질 분석 및 개선
            valid_chunks = []
            chunk_sizes = []
            merged_count = 0
            
            for i, chunk in enumerate(chunks):
                content = chunk.page_content.strip()
                
                # 최소 길이 체크
                if len(content) < 200:  # 200자 미만은 이전 청크와 병합 시도
                    if valid_chunks:
                        last_chunk = valid_chunks[-1]
                        combined_content = last_chunk.page_content + "\n\n" + content
                        
                        # 병합 후 크기가 적절하면 병합
                        if len(combined_content) <= settings.chunk_size * 1.3:
                            valid_chunks[-1] = Document(
                                page_content=combined_content,
                                metadata={**last_chunk.metadata, 'merged_chunks': True}
                            )
                            merged_count += 1
                            continue
                
                # 유효한 청크로 판단
                if len(content) >= 50:  # 최소 50자 이상
                    # 메타데이터 보강
                    enhanced_metadata = chunk.metadata.copy()
                    enhanced_metadata.update({
                        'chunk_index': len(valid_chunks),
                        'chunk_size': len(content),
                        'chunk_id': f"{chunk.metadata.get('file_name', 'unknown')}_{len(valid_chunks):04d}"
                    })
                    
                    valid_chunks.append(Document(
                        page_content=content,
                        metadata=enhanced_metadata
                    ))
                    chunk_sizes.append(len(content))
            
            if chunk_sizes:
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                min_size = min(chunk_sizes)
                max_size = max(chunk_sizes)
                
                logger.info(f"   ✂️  청킹 완료: {len(valid_chunks)}개 청크 ({chunk_time:.1f}초)")
                logger.info(f"   📏 청크 크기: 평균 {avg_size:.0f}자, 최소 {min_size}자, 최대 {max_size}자")
                if merged_count > 0:
                    logger.info(f"   🔗 병합된 청크: {merged_count}개")
                
                status = f"청킹 성공: {len(valid_chunks)}개 청크 (평균 {avg_size:.0f}자)"
            else:
                status = "청킹 실패: 유효한 청크 없음"
            
            return valid_chunks, status
            
        except Exception as e:
            error_msg = f"문서 청킹 실패: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            logger.debug(f"   📋 상세 오류:\n{traceback.format_exc()}")
            return [], error_msg

def get_file_size_mb(file_path: str) -> float:
    """파일 크기를 MB 단위로 반환"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0.0

def test_embedding_model() -> Tuple[bool, str, dict]:
    """
    임베딩 모델 테스트
    Returns: (success, message, model_info)
    """
    try:
        logger.info("🧪 임베딩 모델 테스트 시작...")
        
        embedding_model = EmbeddingModel()
        model_info = embedding_model.get_model_info()
        
        # 한국어 임베딩 테스트
        test_text = "백제 왕성 몽촌토성 발굴조사 보고서"
        test_start = time.time()
        test_embedding = embedding_model.embed_query(test_text)
        test_time = time.time() - test_start
        
        logger.info(f"✅ 임베딩 모델 테스트 성공:")
        logger.info(f"   🏷️  모델: {model_info['model_name']}")
        logger.info(f"   📐 차원: {model_info['dimension']}")
        logger.info(f"   🇰🇷 한국어 최적화: {model_info.get('is_korean_optimized', True)}")
        logger.info(f"   ⏱️  테스트 응답 시간: {test_time:.2f}초")
        logger.info(f"   🎯 테스트 텍스트: '{test_text}'")
        
        return True, "임베딩 모델 테스트 성공", model_info
        
    except Exception as e:
        error_msg = f"임베딩 모델 테스트 실패: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        if "api_key" in str(e).lower() or "upstage" in str(e).lower():
            logger.error("💡 해결 방법:")
            logger.error("   1. .env 파일에 UPSTAGE_API_KEY 설정 확인")
            logger.error("   2. API 키가 유효한지 확인")
            logger.error("   3. 업스테이지 계정의 사용량 한도 확인")
        
        return False, error_msg, {}

def main():
    """마크다운 파일 전용 벡터 DB 구축"""
    
    print("📚 마크다운 파일 전용 벡터 데이터베이스 구축을 시작합니다...")
    print("🎯 특징:")
    print("   - data/documents 폴더의 .md 파일만 처리")
    print("   - 마크다운 구조 인식 최적화")
    print("   - 한국어 텍스트 처리 최적화")
    print("   - 상세한 진행 상황 추적")
    
    total_start_time = time.time()
    
    # 1. 기존 벡터 DB 삭제 확인
    vector_db_path = "vector_db"
    if os.path.exists(vector_db_path):
        print(f"\n⚠️  기존 벡터 DB가 존재합니다: {vector_db_path}")
        response = input("기존 DB를 삭제하고 새로 만드시겠습니까? (y/N): ").strip().lower()
        
        if response in ['y', 'yes', '예']:
            print(f"📁 기존 벡터 DB 삭제 중...")
            try:
                shutil.rmtree(vector_db_path)
                time.sleep(1)  # 파일 시스템 동기화 대기
                logger.info("✅ 기존 벡터 DB 삭제 완료")
            except Exception as e:
                logger.error(f"❌ 기존 벡터 DB 삭제 실패: {str(e)}")
                return
        else:
            print("❌ 사용자가 삭제를 취소했습니다. 종료합니다.")
            return
    
    # 2. 임베딩 모델 테스트
    print("\n🧪 임베딩 모델 테스트...")
    success, message, model_info = test_embedding_model()
    if not success:
        print(f"❌ {message}")
        return
    
    # 3. 마크다운 프로세서 및 벡터 DB 초기화
    print("\n🔄 마크다운 프로세서 및 벡터 DB 초기화...")
    try:
        md_processor = MarkdownDocumentProcessor()
        vector_db = VectorDatabase()
        logger.info("✅ 컴포넌트 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 초기화 실패: {str(e)}")
        return
    
    # 4. 마크다운 파일 스캔
    documents_path = "data/documents"
    print(f"\n📂 마크다운 파일 스캔: {documents_path}")
    
    if not os.path.exists(documents_path):
        print(f"❌ 문서 디렉토리를 찾을 수 없습니다: {documents_path}")
        return
    
    # 5. 마크다운 파일 목록 수집
    md_files = list(Path(documents_path).glob("*.md"))
    
    if not md_files:
        print(f"❌ 처리할 마크다운 파일을 찾을 수 없습니다: {documents_path}")
        return
    
    print(f"🔍 발견된 마크다운 파일: {len(md_files)}개")
    for md_file in md_files:
        file_size_mb = get_file_size_mb(str(md_file))
        print(f"   📄 {md_file.name} ({file_size_mb:.2f}MB)")
    
    # 파일 크기별 정렬 (작은 파일부터 처리)
    md_files.sort(key=lambda f: get_file_size_mb(str(f)))
    
    # 6. 마크다운 파일별 처리
    print(f"\n📝 마크다운 파일 처리 시작 ({len(md_files)}개 파일)...")
    
    all_chunks = []
    success_count = 0
    error_count = 0
    processing_stats = []
    
    for i, file_path in enumerate(md_files, 1):
        file_size_mb = get_file_size_mb(str(file_path))
        file_start_time = time.time()
        
        print(f"\n📄 [{i:2d}/{len(md_files)}] {file_path.name} ({file_size_mb:.2f}MB)")
        logger.info(f"마크다운 파일 처리 시작: {file_path}")
        
        try:
            # 마크다운 파일 로딩
            documents, load_status = md_processor.load_markdown_file(str(file_path))
            
            if not documents:
                print(f"   ❌ {load_status}")
                error_count += 1
                continue
            
            # 청킹
            chunks, chunk_status = md_processor.chunk_documents(documents)
            
            if not chunks:
                print(f"   ❌ {chunk_status}")
                error_count += 1
                continue
            
            # 추가 메타데이터 보강
            for j, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'processing_time': time.time() - file_start_time,
                    'file_order': i,
                    'total_files': len(md_files)
                })
            
            all_chunks.extend(chunks)
            success_count += 1
            
            file_time = time.time() - file_start_time
            processing_stats.append({
                'file': file_path.name,
                'size_mb': file_size_mb,
                'chunks': len(chunks),
                'time': file_time
            })
            
            print(f"   ✅ 성공: {len(chunks)}개 청크 ({file_time:.1f}초)")
            
        except Exception as e:
            error_count += 1
            file_time = time.time() - file_start_time
            print(f"   ❌ 실패: {str(e)} ({file_time:.1f}초)")
            logger.error(f"마크다운 파일 처리 실패 {file_path}: {str(e)}")
            logger.debug(f"상세 오류:\n{traceback.format_exc()}")
    
    # 7. 처리 결과 요약
    print(f"\n📊 마크다운 파일 처리 완료:")
    print(f"   ✅ 성공: {success_count}개")
    print(f"   ❌ 실패: {error_count}개")
    print(f"   📚 총 청크: {len(all_chunks):,}개")
    
    if not all_chunks:
        print("❌ 처리된 청크가 없어서 벡터 DB를 구축할 수 없습니다.")
        return
    
    # 8. 벡터 DB에 저장
    print(f"\n💾 벡터 데이터베이스 저장 중... ({len(all_chunks):,}개 청크)")
    save_start_time = time.time()
    
    try:
        # 청크를 배치로 나누어 저장 (메모리 효율성)
        batch_size = 30  # 마크다운의 경우 텍스트가 길 수 있으므로 배치 크기 조정
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            batch_start = time.time()
            
            vector_db.add_documents(batch)
            
            batch_time = time.time() - batch_start
            progress = (i + len(batch)) / len(all_chunks) * 100
            print(f"   💾 배치 {i//batch_size + 1}: {len(batch)}개 청크 저장 완료 ({progress:.1f}%, {batch_time:.1f}초)")
        
        save_time = time.time() - save_start_time
        print(f"✅ 벡터 DB 저장 완료! ({save_time:.1f}초)")
        
    except Exception as e:
        print(f"❌ 벡터 DB 저장 실패: {str(e)}")
        logger.error(f"벡터 DB 저장 실패: {str(e)}")
        logger.debug(f"상세 오류:\n{traceback.format_exc()}")
        return
    
    # 9. 마크다운 특화 검색 테스트
    print("\n🔍 마크다운 내용 검색 테스트...")
    test_queries = [
        "몽촌토성 발굴조사",
        "백제 왕성",
        "고고학 유물",
        "남한산성 역사"
    ]
    
    for query in test_queries:
        try:
            test_start = time.time()
            results = vector_db.search(query, k=3)
            test_time = time.time() - test_start
            
            if results:
                print(f"   ✅ '{query}': {len(results)}개 결과 ({test_time:.2f}초)")
                # 첫 번째 결과의 미리보기
                first_result = results[0][0] if isinstance(results[0], tuple) else results[0]
                preview = first_result.page_content[:100].replace('\n', ' ')
                source_file = first_result.metadata.get('file_name', 'unknown')
                print(f"      📄 출처: {source_file}")
                print(f"      📝 미리보기: {preview}...")
            else:
                print(f"   ⚠️ '{query}': 결과 없음")
                
        except Exception as e:
            print(f"   ❌ '{query}': 검색 실패 - {str(e)}")
    
    # 10. 최종 통계 및 요약
    total_time = time.time() - total_start_time
    
    print(f"\n🎉 마크다운 전용 벡터 데이터베이스 구축 완료!")
    print(f"📊 최종 통계:")
    print(f"   ⏱️  총 소요 시간: {total_time:.1f}초")
    print(f"   📁 처리된 파일: {success_count}/{len(md_files)}개")
    print(f"   📚 생성된 청크: {len(all_chunks):,}개")
    print(f"   🤖 임베딩 모델: {model_info['model_name']}")
    print(f"   📐 벡터 차원: {model_info['dimension']}")
    
    # 파일별 상세 통계
    if processing_stats:
        print(f"\n📋 파일별 처리 결과:")
        for stat in processing_stats:
            print(f"   📄 {stat['file']}: {stat['chunks']}개 청크 ({stat['time']:.1f}초)")
        
        avg_time = sum(stat['time'] for stat in processing_stats) / len(processing_stats)
        avg_chunks = sum(stat['chunks'] for stat in processing_stats) / len(processing_stats)
        total_size = sum(stat['size_mb'] for stat in processing_stats)
        
        print(f"\n📈 평균 통계:")
        print(f"   ⏱️  평균 처리 시간: {avg_time:.1f}초/파일")
        print(f"   📚 평균 청크 수: {avg_chunks:.1f}개/파일")
        print(f"   📊 총 파일 크기: {total_size:.2f}MB")
    
    print(f"\n✨ 이제 마크다운 문서 내용에 최적화된 검색이 가능합니다!")
    print(f"💡 사용 방법: python app.py를 실행하여 RAG 시스템을 사용해보세요.")

if __name__ == "__main__":
    main() 