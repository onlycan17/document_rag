#!/usr/bin/env python3
"""
안전한 벡터 데이터베이스 재구축 스크립트 (개선 버전)
- API 재시도 로직으로 429 오류 해결
- 상세한 진행 상황 로깅
- OCR 기능 비활성화로 pickle 오류 방지
- 기본 PyPDFLoader만 사용
- 단계별 오류 처리 및 복구
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.vectorstore import VectorDatabase
from src.embeddings import EmbeddingModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import shutil
import os
import time
import logging
from config import settings
from typing import List, Tuple
import traceback

# 로깅 설정 강화
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafeDocumentLoader:
    """안전한 문서 로더 (OCR 없음, 상세 로깅 포함)"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", "。", ".", "!", "?", ";", "；", ",", "，", " ", ""]
        )
        logger.info(f"📝 텍스트 분할기 초기화: 청크 크기={settings.chunk_size}, 겹침={settings.chunk_overlap}")
    
    def load_pdf_safe(self, file_path: str) -> Tuple[List[Document], str]:
        """
        안전한 PDF 로딩 (OCR 없음)
        Returns: (documents, status_message)
        """
        try:
            start_time = time.time()
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            load_time = time.time() - start_time
            
            logger.info(f"   📖 PDF 로딩 완료: {len(documents)}개 페이지 ({load_time:.1f}초)")
            
            # 텍스트 정제
            processed_docs = []
            total_chars_before = 0
            total_chars_after = 0
            
            for i, doc in enumerate(documents):
                original_length = len(doc.page_content)
                total_chars_before += original_length
                
                cleaned_content = self._clean_text(doc.page_content)
                total_chars_after += len(cleaned_content)
                
                if len(cleaned_content.strip()) > 50:  # 최소 길이 체크
                    doc.page_content = cleaned_content
                    processed_docs.append(doc)
                else:
                    logger.debug(f"   ⚠️ 페이지 {i+1} 건너뜀: 텍스트가 너무 짧음 ({original_length}자)")
            
            clean_time = time.time() - start_time - load_time
            logger.info(f"   🧹 텍스트 정제 완료: {total_chars_before:,}자 → {total_chars_after:,}자 ({clean_time:.1f}초)")
            
            status = f"PDF 로딩 성공: {len(processed_docs)}개 유효 페이지"
            return processed_docs, status
            
        except Exception as e:
            error_msg = f"PDF 로딩 실패: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            logger.debug(f"   📋 상세 오류:\n{traceback.format_exc()}")
            return [], error_msg
    
    def load_text_safe(self, file_path: str) -> Tuple[List[Document], str]:
        """
        안전한 텍스트 파일 로딩
        Returns: (documents, status_message)
        """
        try:
            start_time = time.time()
            
            # 다양한 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    documents = loader.load()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"   인코딩 {encoding} 시도 실패: {str(e)}")
                    continue
            
            if not documents:
                return [], "텍스트 파일 로딩 실패: 지원되는 인코딩 없음"
            
            load_time = time.time() - start_time
            logger.info(f"   📝 텍스트 로딩 완료: {used_encoding} 인코딩 ({load_time:.1f}초)")
            
            # 텍스트 정제
            for doc in documents:
                doc.page_content = self._clean_text(doc.page_content)
            
            status = f"텍스트 로딩 성공: {used_encoding} 인코딩"
            return documents, status
            
        except Exception as e:
            error_msg = f"텍스트 파일 로딩 실패: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
            return [], error_msg
    
    def _clean_text(self, text: str) -> str:
        """강화된 텍스트 정제"""
        if not text:
            return ""
        
        import re
        
        # 기본 정제
        text = text.strip()
        
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
        
        return text
    
    def chunk_documents(self, documents: List[Document]) -> Tuple[List[Document], str]:
        """
        문서를 청크로 분할 (상세 통계 포함)
        Returns: (chunks, status_message)
        """
        if not documents:
            return [], "분할할 문서가 없음"
        
        try:
            start_time = time.time()
            chunks = self.text_splitter.split_documents(documents)
            chunk_time = time.time() - start_time
            
            # 청크 품질 분석
            valid_chunks = []
            chunk_sizes = []
            
            for chunk in chunks:
                content = chunk.page_content.strip()
                if content and len(content) > 10:
                    valid_chunks.append(chunk)
                    chunk_sizes.append(len(content))
            
            if chunk_sizes:
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                min_size = min(chunk_sizes)
                max_size = max(chunk_sizes)
                
                logger.info(f"   ✂️  청킹 완료: {len(valid_chunks)}개 청크 ({chunk_time:.1f}초)")
                logger.info(f"   📏 청크 크기: 평균 {avg_size:.0f}자, 최소 {min_size}자, 최대 {max_size}자")
                
                status = f"청킹 성공: {len(valid_chunks)}개 청크 (평균 {avg_size:.0f}자)"
            else:
                status = "청킹 실패: 유효한 청크 없음"
            
            return valid_chunks, status
            
        except Exception as e:
            error_msg = f"문서 청킹 실패: {str(e)}"
            logger.error(f"   ❌ {error_msg}")
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
        
        # 간단한 임베딩 테스트
        test_text = "테스트 임베딩"
        test_start = time.time()
        test_embedding = embedding_model.embed_query(test_text)
        test_time = time.time() - test_start
        
        logger.info(f"✅ 임베딩 모델 테스트 성공:")
        logger.info(f"   🏷️  모델: {model_info['model_name']}")
        logger.info(f"   📐 차원: {model_info['dimension']}")
        logger.info(f"   🇰🇷 한국어 최적화: {model_info.get('is_korean_optimized', False)}")
        logger.info(f"   ⏱️  테스트 응답 시간: {test_time:.2f}초")
        
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
    """개선된 안전한 벡터 DB 재구축"""
    
    print("🚀 안전한 벡터 데이터베이스 재구축을 시작합니다...")
    print("🛡️  개선 사항:")
    print("   - API 재시도 로직으로 429 오류 해결")
    print("   - 상세한 진행 상황 추적")
    print("   - OCR 비활성화로 pickle 오류 방지")
    print("   - 강화된 오류 처리 및 복구")
    
    total_start_time = time.time()
    
    # 1. 기존 벡터 DB 삭제
    vector_db_path = "vector_db"
    if os.path.exists(vector_db_path):
        print(f"\n📁 기존 벡터 DB 삭제 중: {vector_db_path}")
        try:
            shutil.rmtree(vector_db_path)
            time.sleep(1)  # 파일 시스템 동기화 대기
            logger.info("✅ 기존 벡터 DB 삭제 완료")
        except Exception as e:
            logger.error(f"❌ 기존 벡터 DB 삭제 실패: {str(e)}")
            return
    
    # 2. 임베딩 모델 테스트
    print("\n🧪 임베딩 모델 테스트...")
    success, message, model_info = test_embedding_model()
    if not success:
        print(f"❌ {message}")
        return
    
    # 3. 안전한 로더 및 벡터 DB 초기화
    print("\n🔄 안전한 문서 로더 및 벡터 DB 초기화...")
    try:
        safe_loader = SafeDocumentLoader()
        vector_db = VectorDatabase()
        logger.info("✅ 컴포넌트 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 초기화 실패: {str(e)}")
        return
    
    # 4. 문서 디렉토리 확인
    documents_path = "data/documents"
    print(f"\n📚 문서 디렉토리 스캔: {documents_path}")
    
    if not os.path.exists(documents_path):
        print(f"❌ 문서 디렉토리를 찾을 수 없습니다: {documents_path}")
        return
    
    # 5. 파일 목록 수집
    pdf_files = list(Path(documents_path).glob("*.pdf"))
    txt_files = list(Path(documents_path).glob("*.txt"))
    md_files = list(Path(documents_path).glob("*.md"))
    
    all_files = pdf_files + txt_files + md_files
    
    if not all_files:
        print(f"❌ 처리할 문서를 찾을 수 없습니다: {documents_path}")
        return
    
    print(f"🔍 발견된 파일: PDF {len(pdf_files)}개, TXT {len(txt_files)}개, MD {len(md_files)}개")
    
    # 파일 크기별 정렬 (작은 파일부터 처리)
    all_files.sort(key=lambda f: get_file_size_mb(str(f)))
    
    # 6. 파일별 처리
    print(f"\n📄 문서 처리 시작 ({len(all_files)}개 파일)...")
    
    all_chunks = []
    success_count = 0
    error_count = 0
    processing_stats = []
    
    for i, file_path in enumerate(all_files, 1):
        file_size_mb = get_file_size_mb(str(file_path))
        file_start_time = time.time()
        
        print(f"\n📄 [{i:2d}/{len(all_files)}] {file_path.name} ({file_size_mb:.1f}MB)")
        logger.info(f"파일 처리 시작: {file_path}")
        
        try:
            # 파일 유형별 로딩
            if file_path.suffix.lower() == '.pdf':
                documents, load_status = safe_loader.load_pdf_safe(str(file_path))
            else:
                documents, load_status = safe_loader.load_text_safe(str(file_path))
            
            if not documents:
                print(f"   ❌ {load_status}")
                error_count += 1
                continue
            
            # 청킹
            chunks, chunk_status = safe_loader.chunk_documents(documents)
            
            if not chunks:
                print(f"   ❌ {chunk_status}")
                error_count += 1
                continue
            
            # 메타데이터 보강
            for j, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source_file': file_path.name,
                    'file_path': str(file_path),
                    'file_size_mb': file_size_mb,
                    'chunk_index': j,
                    'total_chunks': len(chunks),
                    'processing_time': time.time() - file_start_time
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
            logger.error(f"파일 처리 실패 {file_path}: {str(e)}")
            logger.debug(f"상세 오류:\n{traceback.format_exc()}")
    
    # 7. 처리 결과 요약
    print(f"\n📊 문서 처리 완료:")
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
        batch_size = 50
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
    
    # 9. 검색 테스트
    print("\n🔍 벡터 DB 검색 테스트...")
    test_queries = ["테스트", "문서", "정보"]
    
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
                print(f"      미리보기: {preview}...")
            else:
                print(f"   ⚠️ '{query}': 결과 없음")
                
        except Exception as e:
            print(f"   ❌ '{query}': 검색 실패 - {str(e)}")
    
    # 10. 최종 통계 및 요약
    total_time = time.time() - total_start_time
    
    print(f"\n🎉 벡터 데이터베이스 재구축 완료!")
    print(f"📊 최종 통계:")
    print(f"   ⏱️  총 소요 시간: {total_time:.1f}초")
    print(f"   📁 처리된 파일: {success_count}/{len(all_files)}개")
    print(f"   📚 생성된 청크: {len(all_chunks):,}개")
    print(f"   🤖 임베딩 모델: {model_info['model_name']}")
    print(f"   📐 벡터 차원: {model_info['dimension']}")
    
    # 성능 통계
    if processing_stats:
        avg_time = sum(stat['time'] for stat in processing_stats) / len(processing_stats)
        avg_chunks = sum(stat['chunks'] for stat in processing_stats) / len(processing_stats)
        print(f"   📈 평균 처리 시간: {avg_time:.1f}초/파일")
        print(f"   📈 평균 청크 수: {avg_chunks:.1f}개/파일")
    
    print(f"\n✨ 이제 향상된 검색 성능을 경험할 수 있습니다!")

if __name__ == "__main__":
    main() 