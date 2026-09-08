#!/usr/bin/env python3
"""
문서를 벡터 데이터베이스에 로드하는 스크립트
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.loaders import DocumentLoader
from src.vectorstore import VectorDatabase
from src.utils.document_processor import DocumentProcessor
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="문서를 벡터 데이터베이스에 로드")
    parser.add_argument("path", help="문서 파일 또는 디렉토리 경로")
    parser.add_argument("--clear", action="store_true", help="기존 데이터베이스 초기화")
    parser.add_argument("--batch-size", type=int, default=10, help="배치 크기")

    args = parser.parse_args()

    # 디렉토리 준비
    DocumentProcessor.prepare_directories()

    # 로더 및 벡터 DB 초기화
    loader = DocumentLoader()
    vector_db = VectorDatabase()

    # 데이터베이스 초기화 옵션
    if args.clear:
        print("기존 벡터 데이터베이스를 초기화합니다...")
        vector_db.clear_database()

    # 문서 로드
    if os.path.isfile(args.path):
        print(f"파일 로드 중: {args.path}")
        documents = loader.load_document(args.path)
        vector_db.add_documents(documents)
        print(f"완료! {len(documents)}개의 청크가 추가되었습니다.")

    elif os.path.isdir(args.path):
        print(f"디렉토리 로드 중: {args.path}")
        documents = loader.load_directory(args.path)

        if documents:
            # 배치 처리
            batches = DocumentProcessor.batch_process_directory(args.path, args.batch_size)

            total_docs = 0
            for i, batch_files in enumerate(batches):
                batch_docs = []
                for file_path in batch_files:
                    try:
                        docs = loader.load_document(file_path)
                        batch_docs.extend(docs)
                    except Exception as e:
                        print(f"파일 로드 실패: {file_path} - {str(e)}")

                if batch_docs:
                    vector_db.add_documents(batch_docs)
                    total_docs += len(batch_docs)
                    print(f"배치 {i+1}/{len(batches)} 완료: {len(batch_docs)}개 청크")

            print(f"\n총 {total_docs}개의 청크가 추가되었습니다.")
        else:
            print("로드할 문서가 없습니다.")

    else:
        print(f"경로를 찾을 수 없습니다: {args.path}")
        return

    # 최종 상태 출력
    doc_count = vector_db.get_document_count()
    print(f"\n현재 벡터 데이터베이스에 저장된 총 문서 청크 수: {doc_count}개")


if __name__ == "__main__":
    main()
