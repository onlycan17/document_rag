#!/usr/bin/env python3
"""
로컬 모델 다운로드 스크립트

이 스크립트는 한국어 지능형 이미지 추출기에 필요한 모델들을 다운로드합니다:
1. Midm-2.0-Base-Instruct (GGUF) - 한국어 텍스트 처리
2. Gemma-3n-E4B - 멀티모달 이미지 분석
"""

import os
import sys
from pathlib import Path
import requests
from typing import Optional
import hashlib
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

# 모델 정보
MODELS = {
    "midm-2.0-korean": {
        "name": "Midm-2.0-Base-Instruct-Q4_K_S.gguf",
        "url": "https://huggingface.co/KT-AI/Midm-2.0-Base-Instruct-gguf/resolve/main/Midm-2.0-Base-Instruct-Q4_K_S.gguf",
        "size": "6.66GB",
        "description": "KT의 한국어-영어 바이링귀 모델 (11.5B, 4비트 양자화)",
        "local_path": models_dir / "korean" / "Midm-2.0-Base-Instruct-Q4_K_S.gguf"
    },
    "gemma-3n-e4b": {
        "name": "gemma-2-2b-it",  # Gemma-3n-E4B의 실제 모델명
        "repo_id": "google/gemma-2-2b-it",
        "model_files": [
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json"
        ],
        "size": "~5GB",
        "description": "Google의 효율적인 2B 멀티모달 모델",
        "local_path": models_dir / "multimodal" / "gemma-2-2b-it"
    }
    ,
    "ax-4.0-vl-light": {
        "repo_id": "skt/A.X-4.0-VL-Light",
        "size": "~8GB",
        "description": "SKT A.X 4.0 VL Light 멀티모달 모델",
        "local_path": models_dir / "multimodal" / "A.X-4.0-VL-Light",
    }
    ,
    "exaone-4.0-32b": {
        "repo_id": "LGAI-EXAONE/EXAONE-4.0-32B",
        "size": "~70GB",
        "description": "EXAONE-4.0-32B 텍스트 생성 모델",
        "local_path": models_dir / "local" / "exaone-4.0-32b"
    }
}


def download_file_with_progress(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """진행 표시와 함께 파일 다운로드"""
    try:
        # 디렉토리 생성
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 이미 존재하는 경우 스킵
        if dest_path.exists():
            logger.info(f"✅ 이미 존재: {dest_path.name}")
            return True
        
        logger.info(f"📥 다운로드 시작: {url}")
        
        # 파일 다운로드
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✅ 다운로드 완료: {dest_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 다운로드 실패: {e}")
        if dest_path.exists():
            dest_path.unlink()  # 부분 다운로드 파일 삭제
        return False


def download_huggingface_model(repo_id: str, local_dir: Path, files: list) -> bool:
    """HuggingFace에서 모델 다운로드"""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 스냅샷 다운로드 시도
        logger.info(f"📥 HuggingFace 모델 다운로드: {repo_id}")
        
        try:
            # snapshot_download로 전체 모델 다운로드
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info(f"✅ 모델 다운로드 완료: {repo_id}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 스냅샷 다운로드 실패, 개별 파일 다운로드 시도: {e}")
            
            # 개별 파일 다운로드
            for file_name in files:
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_name,
                        local_dir=str(local_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True
                    )
                    logger.info(f"  ✅ {file_name}")
                except Exception as file_error:
                    logger.error(f"  ❌ {file_name}: {file_error}")
                    return False
            
            return True
            
    except ImportError:
        logger.error("❌ huggingface-hub가 설치되지 않았습니다. 'pip install huggingface-hub' 실행하세요.")
        return False
    except Exception as e:
        logger.error(f"❌ HuggingFace 다운로드 실패: {e}")
        return False


def download_midm_korean_model() -> bool:
    """Midm-2.0 한국어 모델 다운로드"""
    model_info = MODELS["midm-2.0-korean"]
    
    logger.info(f"\n🇰🇷 한국어 모델 다운로드: {model_info['name']}")
    logger.info(f"   설명: {model_info['description']}")
    logger.info(f"   크기: {model_info['size']}")
    
    return download_file_with_progress(
        model_info["url"],
        model_info["local_path"]
    )


def download_gemma_multimodal_model() -> bool:
    """Gemma 멀티모달 모델 다운로드"""
    model_info = MODELS["gemma-3n-e4b"]
    
    logger.info(f"\n🖼️ 멀티모달 모델 다운로드: {model_info['name']}")
    logger.info(f"   설명: {model_info['description']}")
    logger.info(f"   크기: {model_info['size']}")
    
    return download_huggingface_model(
        model_info["repo_id"],
        model_info["local_path"],
        model_info["model_files"]
    )


def download_ax_vl_light_model() -> bool:
    """A.X 4.0 VL Light 모델 다운로드"""
    info = MODELS["ax-4.0-vl-light"]
    logger.info(f"\n🖼️ A.X 4.0 VL Light 다운로드: {info['repo_id']}")
    logger.info(f"   설명: {info['description']}")
    logger.info(f"   크기: {info['size']}")
    return download_huggingface_model(
        info["repo_id"],
        info["local_path"],
        files=[],
    )


def download_exaone_model() -> bool:
    """EXAONE 4.0 32B 다운로드"""
    info = MODELS["exaone-4.0-32b"]
    logger.info(f"\n🧠 EXAONE 모델 다운로드: {info['repo_id']}")
    logger.info(f"   설명: {info['description']}")
    logger.info(f"   크기: {info['size']}")
    return download_huggingface_model(
        info["repo_id"],
        info["local_path"],
        files=[],
    )


def verify_models() -> bool:
    """다운로드된 모델 검증"""
    logger.info("\n🔍 모델 검증 중...")
    
    all_valid = True
    
    # Midm-2.0 검증
    midm_path = MODELS["midm-2.0-korean"]["local_path"]
    if midm_path.exists():
        size_mb = midm_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Midm-2.0 모델: {size_mb:.1f} MB")
    else:
        logger.error(f"❌ Midm-2.0 모델이 없습니다: {midm_path}")
        all_valid = False
    
    # A.X 4.0 VL Light 검증
    ax_dir = MODELS["ax-4.0-vl-light"]["local_path"]
    if ax_dir.exists() and (ax_dir / "config.json").exists():
        files = list(ax_dir.glob("*"))
        logger.info(f"✅ A.X 4.0 VL Light 모델: {len(files)} 파일")
    else:
        logger.error(f"❌ A.X 4.0 VL Light 모델이 없습니다: {ax_dir}")
        all_valid = False
    
    return all_valid


def main():
    """메인 함수"""
    logger.info("=" * 60)
    logger.info("🚀 한국어 지능형 이미지 추출기 모델 다운로드")
    logger.info("=" * 60)
    
    # 모델 디렉토리 정보
    logger.info(f"\n📁 모델 디렉토리: {models_dir}")
    
    # 사용자 확인
    print("\n다음 모델들을 다운로드합니다:")
    print("1. Midm-2.0-Base-Instruct (6.66GB) - 한국어 텍스트 처리")
    print("2. A.X-4.0-VL-Light (~8GB) - 멀티모달 이미지 분석")
    print("3. EXAONE-4.0-32B (~70GB) - 대형 텍스트 생성 모델")
    print(f"\n총 필요 공간: 약 80GB 이상")
    
    response = input("\n계속하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        logger.info("다운로드 취소됨")
        return
    
    # 모델 다운로드
    success = True
    
    # 1. 한국어 모델
    if not download_midm_korean_model():
        success = False
    
    # 2. 멀티모달 모델 (A.X 4.0 VL Light)
    if not download_ax_vl_light_model():
        success = False
    # 3. EXAONE 모델
    if not download_exaone_model():
        success = False
    
    # 모델 검증
    if success:
        if verify_models():
            logger.info("\n✅ 모든 모델이 성공적으로 다운로드되었습니다!")
        else:
            logger.warning("\n⚠️ 일부 모델 검증 실패. 다시 실행하세요.")
    else:
        logger.error("\n❌ 모델 다운로드 중 오류가 발생했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n다운로드가 중단되었습니다.")
        sys.exit(0)