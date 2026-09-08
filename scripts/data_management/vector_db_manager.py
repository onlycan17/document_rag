#!/usr/bin/env python3
"""
통합 벡터 데이터베이스 관리 스크립트

이 스크립트는 기존의 여러 벡터 DB 재구축 스크립트들을 통합하여
사용자가 필요에 따라 다양한 옵션을 선택할 수 있도록 합니다.

주요 기능:
- 전체 문서 처리 (PDF, TXT, MD)
- 마크다운 파일만 처리
- 안전 모드 (OCR 비활성화)
- 특정 문서만 선별 처리
- 상세한 진행 상황 추적
- API 재시도 로직
"""

import sys
import os
import argparse
import time
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Dict

# 프로젝트 루트 경로 설정
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from src.vectorstore import VectorDatabase
from src.embeddings import EmbeddingModel
from src.loaders import DocumentLoader
from src.utils.document_processor import DocumentProcessor
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/vector_db_rebuild.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)


class VectorDBManager:
    """벡터 데이터베이스 통합 관리 클래스"""

    def __init__(self, safe_mode: bool = False, use_ocr: bool = True):
        """
        벡터 DB 관리자 초기화

        Args:
            safe_mode (bool): 안전 모드 (OCR 비활성화, 기본 로더만 사용)
            use_ocr (bool): OCR 사용 여부
        """
        self.safe_mode = safe_mode
        self.use_ocr = use_ocr and not safe_mode
        self.vector_db = None
        self.document_loader = None

        logger.info(f"벡터 DB 관리자 초기화 - 안전모드: {safe_mode}, OCR: {self.use_ocr}")

    def test_embedding_model(self) -> Tuple[bool, str, Dict]:
        """임베딩 모델 테스트"""
        try:
            logger.info("🧪 임베딩 모델 테스트 중...")

            embedding_model = EmbeddingModel()
            test_text = "테스트 문장입니다."

            # 임베딩 생성 테스트
            embeddings = embedding_model.embed_query(test_text)

            model_info = {
                "dimension": len(embeddings),
                "model_type": type(embedding_model.embeddings).__name__,
                "test_successful": True,
            }

            logger.info(f"✅ 임베딩 모델 테스트 성공 - 차원: {model_info['dimension']}")
            return True, "임베딩 모델이 정상적으로 작동합니다", model_info

        except Exception as e:
            error_msg = f"임베딩 모델 테스트 실패: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {}

    def clear_existing_vector_db(self, vector_db_path: str = "vector_db") -> bool:
        """기존 벡터 DB 삭제"""
        try:
            if os.path.exists(vector_db_path):
                logger.info(f"📁 기존 벡터 DB 삭제 중: {vector_db_path}")
                shutil.rmtree(vector_db_path)
                time.sleep(1)  # 파일 시스템 동기화 대기
                logger.info("✅ 기존 벡터 DB 삭제 완료")
            return True
        except Exception as e:
            logger.error(f"❌ 벡터 DB 삭제 실패: {str(e)}")
            return False

    def initialize_components(self) -> bool:
        """필요한 컴포넌트들 초기화"""
        try:
            logger.info("🔄 컴포넌트 초기화 중...")

            # 디렉토리 준비
            DocumentProcessor.prepare_directories()

            # 벡터 DB 초기화
            self.vector_db = VectorDatabase()

            # 문서 로더 초기화
            if self.safe_mode:
                logger.info("🛡️ 안전 모드: 기본 로더만 사용")
                self.document_loader = None  # 직접 PyPDFLoader 사용
            else:
                self.document_loader = DocumentLoader(use_ocr=self.use_ocr)

            logger.info("✅ 컴포넌트 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 컴포넌트 초기화 실패: {str(e)}")
            return False

    def get_file_list(self, documents_dir: str, file_types: List[str] = None) -> List[Path]:
        """처리할 파일 목록 수집"""
        if file_types is None:
            file_types = ["*.pdf", "*.txt", "*.md"]

        doc_path = Path(documents_dir)
        if not doc_path.exists():
            logger.error(f"❌ 문서 디렉토리를 찾을 수 없습니다: {documents_dir}")
            return []

        files = []
        for pattern in file_types:
            files.extend(doc_path.glob(pattern))

        # 숨김 파일 제외
        files = [f for f in files if not f.name.startswith(".")]

        logger.info(f"📚 발견된 파일: {len(files)}개")
        return sorted(files)

    def process_file_safe(self, file_path: Path) -> Tuple[List[Document], str]:
        """안전 모드로 파일 처리"""
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
            elif file_path.suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents = loader.load()
            else:
                return [], f"지원하지 않는 파일 형식: {file_path.suffix}"

            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, length_function=len
            )

            split_documents = text_splitter.split_documents(documents)
            return split_documents, "성공"

        except Exception as e:
            error_msg = f"파일 처리 실패: {str(e)}"
            logger.error(f"❌ {file_path.name}: {error_msg}")
            return [], error_msg

    def process_file_advanced(self, file_path: Path) -> Tuple[List[Document], str]:
        """고급 모드로 파일 처리 (OCR 포함)"""
        try:
            documents = self.document_loader.load_document(str(file_path))
            return documents, "성공"
        except Exception as e:
            error_msg = f"파일 처리 실패: {str(e)}"
            logger.error(f"❌ {file_path.name}: {error_msg}")
            return [], error_msg

    def rebuild_vector_db(
        self,
        documents_dir: str = "data/documents",
        file_types: List[str] = None,
        markdown_only: bool = False,
        specific_files: List[str] = None,
    ) -> Dict:
        """
        벡터 데이터베이스 재구축

        Args:
            documents_dir: 문서 디렉토리 경로
            file_types: 처리할 파일 형식 목록
            markdown_only: 마크다운 파일만 처리
            specific_files: 특정 파일들만 처리

        Returns:
            Dict: 처리 결과 정보
        """
        start_time = time.time()

        # 파일 형식 설정
        if markdown_only:
            file_types = ["*.md"]
            logger.info("📝 마크다운 파일만 처리합니다")
        elif file_types is None:
            file_types = ["*.pdf", "*.txt", "*.md"]

        # 1. 임베딩 모델 테스트
        success, message, model_info = self.test_embedding_model()
        if not success:
            return {"success": False, "error": message}

        # 2. 기존 벡터 DB 삭제
        if not self.clear_existing_vector_db():
            return {"success": False, "error": "기존 벡터 DB 삭제 실패"}

        # 3. 컴포넌트 초기화
        if not self.initialize_components():
            return {"success": False, "error": "컴포넌트 초기화 실패"}

        # 4. 파일 목록 수집
        if specific_files:
            files = [Path(documents_dir) / file for file in specific_files]
            files = [f for f in files if f.exists()]
        else:
            files = self.get_file_list(documents_dir, file_types)

        if not files:
            return {"success": False, "error": "처리할 파일이 없습니다"}

        # 5. 파일별 처리
        logger.info(f"🚀 {len(files)}개 파일 처리 시작")

        stats = {
            "total_files": len(files),
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "processing_time": 0,
            "errors": [],
        }

        for i, file_path in enumerate(files, 1):
            logger.info(f"📄 [{i}/{len(files)}] {file_path.name} 처리 중...")

            try:
                # 파일 처리
                if self.safe_mode:
                    documents, status = self.process_file_safe(file_path)
                else:
                    documents, status = self.process_file_advanced(file_path)

                if documents:
                    # 벡터 DB에 추가
                    self.vector_db.add_documents(documents)
                    stats["processed_files"] += 1
                    stats["total_chunks"] += len(documents)
                    logger.info(f"✅ {file_path.name}: {len(documents)}개 청크 처리 완료")
                else:
                    stats["failed_files"] += 1
                    stats["errors"].append(f"{file_path.name}: {status}")
                    logger.warning(f"⚠️ {file_path.name}: 처리된 문서가 없습니다")

            except Exception as e:
                stats["failed_files"] += 1
                error_msg = f"{file_path.name}: {str(e)}"
                stats["errors"].append(error_msg)
                logger.error(f"❌ {error_msg}")

        # 6. 결과 정리
        stats["processing_time"] = time.time() - start_time
        stats["success"] = stats["processed_files"] > 0

        # 벡터 DB 상태 확인
        if self.vector_db:
            stats["final_document_count"] = self.vector_db.get_document_count()

        return stats

    def generate_report(self, stats: Dict) -> str:
        """처리 결과 보고서 생성"""
        report = [
            "=" * 60,
            "🎉 벡터 데이터베이스 재구축 완료!",
            "=" * 60,
            "📊 처리 통계:",
            f"  • 전체 파일: {stats['total_files']}개",
            f"  • 성공: {stats['processed_files']}개",
            f"  • 실패: {stats['failed_files']}개",
            f"  • 총 청크: {stats['total_chunks']}개",
            f"  • 처리 시간: {stats['processing_time']:.1f}초",
            f"  • 최종 DB 문서 수: {stats.get('final_document_count', 'N/A')}개",
            "",
        ]

        if stats.get("errors"):
            report.extend(["❌ 오류 목록:", *[f"  • {error}" for error in stats["errors"]], ""])

        report.extend(["✨ 다음 명령어로 앱을 실행하세요:", "  streamlit run app.py", ""])

        return "\n".join(report)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="통합 벡터 데이터베이스 관리 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 문서 처리 (기본)
  python vector_db_manager.py
  
  # 안전 모드 (OCR 비활성화)
  python vector_db_manager.py --safe-mode
  
  # 마크다운 파일만 처리
  python vector_db_manager.py --markdown-only
  
  # 특정 파일들만 처리
  python vector_db_manager.py --files "file1.pdf" "file2.md"
  
  # OCR 비활성화
  python vector_db_manager.py --no-ocr
        """,
    )

    parser.add_argument("--safe-mode", action="store_true", help="안전 모드 (OCR 비활성화, 기본 로더만 사용)")
    parser.add_argument("--no-ocr", action="store_true", help="OCR 사용 안함")
    parser.add_argument("--markdown-only", action="store_true", help="마크다운 파일만 처리")
    parser.add_argument("--documents-dir", default="data/documents", help="문서 디렉토리 경로 (기본: data/documents)")
    parser.add_argument("--files", nargs="+", help="처리할 특정 파일들 (공백으로 구분)")
    parser.add_argument("--types", nargs="+", choices=["pdf", "txt", "md"], help="처리할 파일 형식 선택")

    args = parser.parse_args()

    # 파일 형식 설정
    file_types = None
    if args.types:
        file_types = [f"*.{ext}" for ext in args.types]

    print("🚀 통합 벡터 데이터베이스 관리 도구")
    print("=" * 50)

    try:
        # 관리자 인스턴스 생성
        manager = VectorDBManager(safe_mode=args.safe_mode, use_ocr=not args.no_ocr)

        # 벡터 DB 재구축
        stats = manager.rebuild_vector_db(
            documents_dir=args.documents_dir,
            file_types=file_types,
            markdown_only=args.markdown_only,
            specific_files=args.files,
        )

        # 결과 보고서 출력
        report = manager.generate_report(stats)
        print(report)

        # 로그 파일에도 저장
        with open("logs/rebuild_summary.log", "w", encoding="utf-8") as f:
            f.write(report)

        # 종료 코드 설정
        exit_code = 0 if stats["success"] else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류 발생: {str(e)}")
        print(f"❌ 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
