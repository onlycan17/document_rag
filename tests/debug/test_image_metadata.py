"""
벡터 DB에 저장된 이미지 메타데이터 확인 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstore import VectorDatabase


def test_image_metadata():
    """벡터 DB에 저장된 문서의 이미지 메타데이터 확인"""

    print("=" * 80)
    print("벡터 DB 이미지 메타데이터 확인 테스트")
    print("=" * 80)

    # 1. 벡터 DB 초기화
    print("\n1. 벡터 DB 연결 중...")
    vector_db = VectorDatabase()

    if vector_db.get_document_count() == 0:
        print("❌ 벡터 DB가 초기화되지 않았습니다.")
        return

    print(f"✅ 벡터 DB 연결 완료: {vector_db.get_document_count()}개 문서")

    # 2. 몽촌토성 관련 문서 검색
    print("\n2. '몽촌토성' 검색 중...")
    query = "몽촌토성"
    results = vector_db.search(query, k=5)

    print(f"\n검색 결과: {len(results)}개 문서")

    # 3. 각 문서의 메타데이터 확인
    print("\n3. 문서별 이미지 메타데이터 확인:")
    print("-" * 80)

    image_fields = [
        "image_paths",
        "image_files",
        "images",
        "intelligent_images",
        "image_path",
        "image_file",
        "relative_path",
        "image_description",
    ]

    for i, (doc, _score) in enumerate(results, 1):
        print(f"\n[문서 {i}]")

        # 메타데이터 접근
        metadata = {}
        if isinstance(doc, dict):
            metadata = doc.get("metadata", doc)
        else:
            metadata = getattr(doc, "metadata", {}) or {}

        # 기본 정보
        print(f"  출처: {metadata.get('source', 'Unknown')}")
        print(f"  파일명: {metadata.get('file_name', 'Unknown')}")

        # 이미지 관련 필드 확인
        has_image = False
        for field in image_fields:
            value = metadata.get(field)
            if value:
                has_image = True
                print(f"  {field}: {value}")

        if not has_image:
            print("  ⚠️ 이미지 관련 메타데이터 없음")

        # 전체 메타데이터 키 출력
        print(f"  메타데이터 키: {list(metadata.keys())}")

    # 4. 결과 요약
    print("\n" + "=" * 80)
    print("결과 요약:")
    print("=" * 80)

    total_docs = len(results)
    docs_with_images = sum(
        1
        for doc, _score in results
        if any(
            (metadata := getattr(doc, "metadata", {}) if not isinstance(doc, dict) else doc.get("metadata", {}))
            and metadata.get(field)
            for field in image_fields
        )
    )

    print(f"총 문서: {total_docs}개")
    print(f"이미지 메타데이터 포함: {docs_with_images}개")
    print(f"이미지 메타데이터 없음: {total_docs - docs_with_images}개")

    if docs_with_images == 0:
        print("\n❌ 문제 발견: 검색된 문서에 이미지 메타데이터가 없습니다!")
        print("   → 문서 로딩 시 이미지 메타데이터가 저장되지 않았을 가능성")
    else:
        print(f"\n✅ {docs_with_images}개 문서에 이미지 메타데이터가 있습니다")


if __name__ == "__main__":
    test_image_metadata()
