#!/usr/bin/env python3
"""
벡터 데이터베이스를 완전히 재구축하는 스크립트
- 기존 벡터 DB 삭제
- 개선된 청크 설정으로 문서 재로드
- 품질 검증
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.loaders import DocumentLoader
from src.vectorstore import VectorDatabase
from src.utils.document_processor import DocumentProcessor
import shutil
import os
import time

def main():
    """벡터 DB 재구축"""
    
    print("🔧 벡터 데이터베이스 재구축을 시작합니다...")
    
    # 1. 기존 벡터 DB 삭제
    vector_db_path = "vector_db"
    if os.path.exists(vector_db_path):
        print(f"📁 기존 벡터 DB 삭제 중: {vector_db_path}")
        shutil.rmtree(vector_db_path)
        time.sleep(1)  # 파일 시스템이 완전히 정리될 때까지 대기
    
    # 2. 디렉토리 준비
    DocumentProcessor.prepare_directories()
    
    # 3. 새로운 로더 및 벡터 DB 초기화
    print("🔄 새로운 문서 로더 및 벡터 DB 초기화 중...")
    loader = DocumentLoader()
    vector_db = VectorDatabase()
    
    # 4. 문서 디렉토리 로드
    documents_path = "data/documents"
    print(f"📚 문서 로드 중: {documents_path}")
    
    if not os.path.exists(documents_path):
        print(f"❌ 문서 디렉토리를 찾을 수 없습니다: {documents_path}")
        return
    
    # 5. 문서별 개별 처리 (진행상황 표시)
    total_docs = 0
    processed_files = 0
    failed_files = 0
    
    # PDF 파일 목록 가져오기
    pdf_files = []
    for file in os.listdir(documents_path):
        if file.endswith('.pdf') and not file.startswith('.'):
            pdf_files.append(os.path.join(documents_path, file))
    
    print(f"🔍 발견된 PDF 파일: {len(pdf_files)}개")
    
    for i, file_path in enumerate(pdf_files, 1):
        file_name = os.path.basename(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        print(f"\n📄 [{i}/{len(pdf_files)}] {file_name} ({file_size_mb:.1f}MB)")
        
        try:
            start_time = time.time()
            
            # 문서 로드
            documents = loader.load_document(file_path)
            
            if documents:
                # 벡터 DB에 추가
                vector_db.add_documents(documents)
                
                processing_time = time.time() - start_time
                total_docs += len(documents)
                processed_files += 1
                
                print(f"   ✅ 성공: {len(documents)}개 청크 생성 (처리시간: {processing_time:.1f}초)")
                
                # 청크 크기 분석
                chunk_sizes = [len(doc.page_content) for doc in documents]
                if chunk_sizes:
                    avg_size = sum(chunk_sizes) / len(chunk_sizes)
                    min_size = min(chunk_sizes)
                    max_size = max(chunk_sizes)
                    print(f"   📊 청크 크기: 평균 {avg_size:.0f}자, 최소 {min_size}자, 최대 {max_size}자")
            else:
                failed_files += 1
                print(f"   ❌ 실패: 문서를 로드할 수 없음")
                
        except Exception as e:
            failed_files += 1
            print(f"   ❌ 실패: {str(e)}")
    
    # 6. 최종 결과 출력
    print(f"\n🎉 벡터 DB 재구축 완료!")
    print(f"   📈 처리된 파일: {processed_files}개")
    print(f"   ❌ 실패한 파일: {failed_files}개")
    print(f"   📚 총 청크 수: {total_docs}개")
    
    # 7. 품질 검증
    print(f"\n🔍 품질 검증 중...")
    final_doc_count = vector_db.get_document_count()
    print(f"   💾 저장된 청크 수: {final_doc_count}개")
    
    if final_doc_count != total_docs:
        print(f"   ⚠️  경고: 로드된 청크 수({total_docs})와 저장된 청크 수({final_doc_count})가 다릅니다.")
    
    # 8. 간단한 검색 테스트
    print(f"\n🎯 검색 테스트...")
    test_queries = ["몽촌토성", "백제", "발굴조사"]
    for query in test_queries:
        results = vector_db.search(query, k=3)
        print(f"   '{query}': {len(results)}개 결과")
        if results:
            avg_length = sum(len(doc.page_content) for doc, _ in results) / len(results)
            print(f"   평균 콘텐츠 길이: {avg_length:.0f}자")
    
    print(f"\n✨ 재구축 완료! 이제 더 나은 품질의 답변을 받을 수 있습니다.")

if __name__ == "__main__":
    main() 