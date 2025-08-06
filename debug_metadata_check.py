#!/usr/bin/env python3
"""
벡터 DB 메타데이터 구조 디버깅
- FAISS 인덱스에서 실제 문서 메타데이터 확인
- 이미지 정보가 올바르게 저장되었는지 검증
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings.embedding_model import EmbeddingModel
from src.rag.rag_chain import RAGChain
import json

def check_document_metadata():
    """문서 메타데이터 구조 확인"""
    print("🔍 벡터 DB 메타데이터 구조 확인")
    print("=" * 50)
    
    try:
        # RAG 체인 초기화
        rag = RAGChain()
        
        # 검색 테스트
        query = "몽촌토성 유물"
        
        print(f"검색어: {query}")
        print("-" * 30)
        
        # 검색 실행
        retriever = rag.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        docs = retriever.get_relevant_documents(query)
        
        print(f"검색된 문서 수: {len(docs)}")
        print()
        
        for i, doc in enumerate(docs[:3], 1):  # 처음 3개만 확인
            print(f"📄 문서 {i}:")
            print(f"   내용 길이: {len(doc.page_content)}자")
            print(f"   메타데이터 키: {list(doc.metadata.keys())}")
            
            # 상세 메타데이터 출력
            metadata = doc.metadata
            for key, value in metadata.items():
                if key == 'images' and isinstance(value, list) and len(value) > 0:
                    print(f"   🖼️ {key}: {len(value)}개 이미지")
                    for j, img in enumerate(value[:2]):  # 처음 2개만
                        if isinstance(img, dict):
                            print(f"      - 이미지 {j+1}: {list(img.keys())}")
                        else:
                            print(f"      - 이미지 {j+1}: {type(img)}")
                elif key == 'source':
                    print(f"   📁 {key}: {Path(str(value)).name}")
                elif key in ['page', 'chunk_index']:
                    print(f"   📑 {key}: {value}")
                else:
                    print(f"   📋 {key}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
            print()
        
        # 이미지가 있는 문서 찾기
        docs_with_images = []
        for doc in docs:
            if 'images' in doc.metadata and doc.metadata['images']:
                docs_with_images.append(doc)
        
        print(f"🖼️ 이미지가 포함된 문서: {len(docs_with_images)}개")
        
        if docs_with_images:
            print("첫 번째 이미지 정보 상세:")
            first_img_doc = docs_with_images[0]
            images = first_img_doc.metadata.get('images', [])
            if images:
                first_img = images[0]
                if isinstance(first_img, dict):
                    print(json.dumps(first_img, indent=2, ensure_ascii=False))
                else:
                    print(f"이미지 데이터 타입: {type(first_img)}")
                    print(f"이미지 내용: {str(first_img)[:200]}...")
        else:
            print("⚠️ 이미지가 포함된 문서를 찾을 수 없습니다!")
            
        return docs_with_images
        
    except Exception as e:
        print(f"❌ 메타데이터 확인 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def test_image_display_logic():
    """이미지 표시 로직 테스트"""
    print("\n🖼️ 이미지 표시 로직 테스트")
    print("=" * 50)
    
    try:
        # RAG 체인 초기화
        rag = RAGChain()
        
        # 테스트 질문
        question = "몽촌토성에서 출토된 유물들을 보여줘"
        print(f"질문: {question}")
        
        # RAG 응답 생성
        response = rag({"question": question})
        
        print(f"\n응답 길이: {len(response['answer'])}자")
        print(f"소스 문서 수: {len(response.get('source_documents', []))}")
        
        # 소스 문서들의 이미지 정보 확인
        source_docs = response.get('source_documents', [])
        total_images = 0
        
        for i, doc in enumerate(source_docs):
            images = doc.metadata.get('images', [])
            if images:
                total_images += len(images)
                print(f"문서 {i+1}: {len(images)}개 이미지")
        
        print(f"전체 이미지 수: {total_images}개")
        
        # 응답에 "### 📸 관련 이미지" 섹션이 있는지 확인
        has_image_section = "### 📸 관련 이미지" in response['answer']
        print(f"이미지 섹션 포함: {'✅' if has_image_section else '❌'}")
        
        if has_image_section:
            # 이미지 섹션 내용 확인
            img_section_start = response['answer'].find("### 📸 관련 이미지")
            img_section_content = response['answer'][img_section_start:img_section_start + 500]
            print(f"이미지 섹션 미리보기:\n{img_section_content}")
        else:
            print("⚠️ 응답에 이미지 섹션이 없습니다!")
            
            # display_images_in_response 함수를 직접 호출해보기
            print("\n🔧 이미지 표시 함수 직접 테스트:")
            from app import display_images_in_response
            
            enhanced_response = display_images_in_response(response['answer'], source_docs)
            has_enhanced_images = "### 📸 관련 이미지" in enhanced_response
            
            print(f"수동 이미지 추가 후: {'✅' if has_enhanced_images else '❌'}")
            
            if has_enhanced_images:
                print("✅ 이미지 표시 함수는 정상 작동함")
                return True
            else:
                print("❌ 이미지 표시 함수도 작동하지 않음")
                return False
        
        return has_image_section
        
    except Exception as e:
        print(f"❌ 이미지 표시 로직 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 벡터 DB 메타데이터 디버깅 시작")
    print()
    
    # 1. 메타데이터 구조 확인
    docs_with_images = check_document_metadata()
    
    # 2. 이미지 표시 로직 테스트
    image_display_works = test_image_display_logic()
    
    print("\n📊 디버깅 결과:")
    print(f"이미지 메타데이터 존재: {'✅' if docs_with_images else '❌'}")
    print(f"이미지 표시 기능: {'✅' if image_display_works else '❌'}")
    
    if not docs_with_images:
        print("\n💡 해결 방안:")
        print("1. 문서 재처리로 이미지 메타데이터 생성")
        print("2. 이미지 추출 기능 활성화 확인")
        print("3. PDF 변환 시 이미지 처리 로직 점검")