#!/usr/bin/env python3
"""
몽촌토성 관련 주요 문서들만 선별적으로 처리하는 스크립트
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
    """주요 몽촌토성 문서만 처리"""
    
    print("🏛️ 몽촌토성 관련 주요 문서만 처리합니다...")
    
    # 처리할 주요 파일들 (몽촌토성 관련)
    key_files = [
        "몽촌토성+북묵지+내측+발굴조사+보고서+1.pdf",
        "몽촌토성4+상.pdf", 
        "몽촌토성4+하.pdf",
        "2021년+몽촌토성+북문지+일원+발굴조사+자료집.pdf",
        "c16de876192737d9664a7372b9cd94b8.pdf",  # 이미 성공한 파일
        "000000169391.pdf",
        "0712 백제 왕궁·왕성(백제왕도 핵심유적, 풍납·몽촌토성 등) 연구 성과 발표(붙임).pdf"
    ]
    
    # 1. 기존 벡터 DB 삭제
    vector_db_path = "vector_db"
    if os.path.exists(vector_db_path):
        print(f"📁 기존 벡터 DB 삭제 중...")
        shutil.rmtree(vector_db_path)
        time.sleep(1)
    
    # 2. 초기화
    DocumentProcessor.prepare_directories()
    loader = DocumentLoader()
    vector_db = VectorDatabase()
    
    # 3. 주요 파일들만 처리
    documents_path = "data/documents"
    total_docs = 0
    processed_files = 0
    
    for i, file_name in enumerate(key_files, 1):
        file_path = os.path.join(documents_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"⏭️  [{i}/{len(key_files)}] {file_name} - 파일 없음, 건너뜀")
            continue
            
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"\n📄 [{i}/{len(key_files)}] {file_name} ({file_size_mb:.1f}MB)")
        
        try:
            start_time = time.time()
            documents = loader.load_document(file_path)
            
            if documents:
                vector_db.add_documents(documents)
                processing_time = time.time() - start_time
                total_docs += len(documents)
                processed_files += 1
                
                # 청크 크기 분석
                chunk_sizes = [len(doc.page_content) for doc in documents]
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                min_size = min(chunk_sizes)
                max_size = max(chunk_sizes)
                
                print(f"   ✅ 성공: {len(documents)}개 청크 ({processing_time:.1f}초)")
                print(f"   📊 크기: 평균 {avg_size:.0f}자, 최소 {min_size}자, 최대 {max_size}자")
            else:
                print(f"   ❌ 문서 로드 실패")
                
        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")
    
    # 4. 결과 출력
    print(f"\n🎉 주요 문서 처리 완료!")
    print(f"   📈 처리된 파일: {processed_files}개")
    print(f"   📚 총 청크 수: {total_docs}개")
    
    # 5. 품질 검증
    final_doc_count = vector_db.get_document_count()
    print(f"   💾 저장된 청크: {final_doc_count}개")
    
    # 6. 몽촌토성 검색 테스트
    print(f"\n🔍 몽촌토성 검색 테스트...")
    results = vector_db.search("몽촌토성 발굴조사", k=5)
    print(f"   🎯 검색 결과: {len(results)}개")
    
    if results:
        for i, (doc, score) in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            content_length = len(doc.page_content)
            print(f"   {i}. {source} (길이: {content_length}자, 점수: {score:.3f})")
            print(f"      내용: {doc.page_content[:100].replace('\\n', ' ')}...")
    
    print(f"\n✨ 몽촌토성 전용 벡터 DB 구축 완료!")

if __name__ == "__main__":
    main() 