from src.utils.md_postprocessor import MDPostProcessor
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/postprocess_manual.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

try:
    logger.info("🚀 수동 후처리 시작")
    processor = MDPostProcessor()
    input_file = 'converted_docs/몽촌토성4+하.md'
    output_file = processor.process_md_file(input_file)
    logger.info(f'✅ 후처리 완료: {output_file}')
    print(f'SUCCESS: {output_file}')
except Exception as e:
    logger.error(f'❌ 후처리 실패: {e}', exc_info=True)
    print(f'ERROR: {e}')
    sys.exit(1)
