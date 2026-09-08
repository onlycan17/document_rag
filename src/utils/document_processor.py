import os
from typing import List, Dict
from pathlib import Path


class DocumentProcessor:
    """문서 처리를 위한 유틸리티 클래스"""

    @staticmethod
    def prepare_directories():
        """필요한 디렉토리 생성"""
        directories = ["./data/documents", "./data/processed", "./vector_db", "./logs"]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 정제"""
        # 불필요한 공백 제거
        text = " ".join(text.split())

        # 특수 문자 정리
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("\ufeff", "")  # BOM

        return text.strip()

    @staticmethod
    def batch_process_directory(directory_path: str, batch_size: int = 10) -> List[List[str]]:
        """디렉토리 내 파일을 배치로 분할"""
        files = []
        supported_extensions = [".txt", ".md", ".pdf"]

        for root, dirs, filenames in os.walk(directory_path):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in supported_extensions):
                    files.append(os.path.join(root, filename))

        # 배치로 분할
        batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]
        return batches

    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, any]:
        """파일 정보 추출"""
        path = Path(file_path)
        return {
            "name": path.name,
            "extension": path.suffix,
            "size": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "modified": path.stat().st_mtime,
        }
