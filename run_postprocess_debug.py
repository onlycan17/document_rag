import os
os.environ['OPENROUTER_API_KEY'] = 'REDACTED_OPENROUTER_KEY'

from src.utils.md_postprocessor import MDPostProcessor
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/postprocess_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

try:
    logger.info("🚀 수동 후처리 시작 (DEBUG 모드)")
    logger.info(f"API Key 설정: {os.environ.get('OPENROUTER_API_KEY', 'NOT SET')[:20]}...")
    
    processor = MDPostProcessor()
    logger.info(f"후처리 모델: {processor.model_name}")
    logger.info(f"제공자: {processor.provider}")
    
    input_file = 'converted_docs/몽촌토성4+하.md'
    logger.info(f"입력 파일: {input_file}")
    
    output_file = processor.process_md_file(input_file)
    logger.info(f'✅ 후처리 완료: {output_file}')
    print(f'SUCCESS: {output_file}')
except Exception as e:
    logger.error(f'❌ 후처리 실패: {e}', exc_info=True)
    print(f'ERROR: {e}')
    sys.exit(1)
