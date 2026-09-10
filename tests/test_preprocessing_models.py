"""
전처리 모델 테스트 스크립트

로컬 및 외부 API 모델을 테스트하여 정상 작동을 확인합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.processing.preprocessing_factory import PreprocessingModelFactory
from src.loaders.document_loader import DocumentLoader
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_preprocessing_models():
    """전처리 모델들을 테스트합니다."""
    logger.info("=== 전처리 모델 테스트 시작 ===")
    
    # 테스트할 모델 목록
    test_models = ["local", "openai", "google", "anthropic"]
    
    # 사용 가능한 모델 확인
    available_models = PreprocessingModelFactory.get_available_models()
    logger.info(f"사용 가능한 모델: {list(available_models.keys())}")
    
    for model_type in test_models:
        logger.info(f"\n--- {model_type.upper()} 모델 테스트 ---")
        
        try:
            # 모델 생성
            model = PreprocessingModelFactory.create_model(model_type)
            logger.info(f"✅ {model_type} 모델 생성 성공")
            
            # 모델 정보 확인
            model_info = model.get_model_info()
            logger.info(f"모델 정보: {model_info}")
            
            # 사용 가능 여부 확인
            is_available = model.is_available()
            logger.info(f"사용 가능: {is_available}")
            
            if is_available:
                # 텍스트 전처리 테스트
                test_text = """
                이것은 테스트 문서입니다.  한국어 문장이 잘 연결되는지 확인합니다.
                PDF에서 추출한 텍스트처럼   불필요한 공백이나  줄바꿈이 있는 경우
                전처리 모델이 이를 개선해야 합니다.
                
                문단이 분리되어 있고,  적절한 처리가 필요합니다.
                """
                
                processed_text = model.preprocess_text(test_text)
                logger.info(f"전처리 전: {len(test_text)}자")
                logger.info(f"전처리 후: {len(processed_text)}자")
                logger.info(f"전처리 결과: {processed_text[:100]}...")
                
                # 품질 검증
                if len(processed_text.strip()) > 0:
                    logger.info("✅ 텍스트 전처리 성공")
                else:
                    logger.warning("⚠️ 전처리 결과가 비어있습니다")
            
        except Exception as e:
            logger.error(f"❌ {model_type} 모델 테스트 실패: {e}")
    
    logger.info("\n=== 전처리 모델 테스트 완료 ===")


def test_document_loader_integration():
    """DocumentLoader와의 통합을 테스트합니다."""
    logger.info("\n=== DocumentLoader 통합 테스트 시작 ===")
    
    # 테스트할 모델
    test_models = ["local"]  # 기본적으로 로컬 모델만 테스트
    
    for model_type in test_models:
        logger.info(f"\n--- DocumentLoader with {model_type.upper()} ---")
        
        try:
            # DocumentLoader 생성
            loader = DocumentLoader(
                use_ocr=False,
                use_agent_preprocessing=False,
                enable_postprocessing=False,
                preprocessing_model=model_type
            )
            
            logger.info(f"✅ DocumentLoader 생성 성공 (모델: {model_type})")
            
            # 전처리 모델이 초기화되었는지 확인
            if loader._preprocessing_model:
                logger.info("✅ 전처리 모델이 초기화되었습니다")
                
                # 모델 정보 확인
                model_info = loader._preprocessing_model.get_model_info()
                logger.info(f"사용 중인 모델: {model_info}")
            else:
                logger.warning("⚠️ 전처리 모델이 초기화되지 않았습니다")
            
        except Exception as e:
            logger.error(f"❌ DocumentLoader 통합 테스트 실패: {e}")
    
    logger.info("\n=== DocumentLoader 통합 테스트 완료 ===")


def test_model_factory():
    """모델 팩토리 기능을 테스트합니다."""
    logger.info("\n=== 모델 팩토리 테스트 시작 ===")
    
    try:
        # 사용 가능한 모델 목록 확인
        available_models = PreprocessingModelFactory.get_available_models()
        logger.info(f"사용 가능한 모델 목록: {available_models}")
        
        # 모델 요구사항 확인
        for model_type in ["local", "openai", "google", "anthropic"]:
            requirements = PreprocessingModelFactory.get_model_requirements(model_type)
            logger.info(f"{model_type} 요구사항: {requirements}")
        
        # 사용 가능 여부 확인
        for model_type in ["local", "openai", "google", "anthropic"]:
            is_available = PreprocessingModelFactory.check_model_availability(model_type)
            logger.info(f"{model_type} 사용 가능: {is_available}")
        
        logger.info("✅ 모델 팩토리 테스트 성공")
        
    except Exception as e:
        logger.error(f"❌ 모델 팩토리 테스트 실패: {e}")
    
    logger.info("\n=== 모델 팩토리 테스트 완료 ===")


def main():
    """메인 테스트 함수"""
    try:
        # 1. 모델 팩토리 테스트
        test_model_factory()
        
        # 2. 전처리 모델 테스트
        test_preprocessing_models()
        
        # 3. DocumentLoader 통합 테스트
        test_document_loader_integration()
        
        logger.info("\n🎉 모든 테스트가 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()