"""
한국어 지능형 이미지 추출기

로컬 모델을 사용하여 PDF 문서에서 지능적으로 이미지를 추출하고 분류합니다:
1. Midm-2.0으로 문서 주제 파악
2. Gemma-2로 이미지 분석 및 관련성 판단
3. 텍스트 이미지는 OCR로 텍스트 추출
4. 관련 이미지만 선별하여 저장
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from datetime import datetime
import json
import unicodedata
import shutil

from .korean_text_model import KoreanTextModel
from .gemma_multimodal import GemmaMultimodalModel
from .text_processing import TextProcessor

logger = logging.getLogger(__name__)


class IntelligentImageExtractorKorean:
    """한국어 지능형 이미지 추출기"""
    
    def __init__(self,
                 output_dir: str = "intelligent_extraction_korean",
                 relevance_threshold: float = 0.6,
                 min_image_size: int = 100,
                 max_image_size: int = 4096,
                 enable_ocr: bool = True,
                 use_local_models: bool = True):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리
            relevance_threshold: 관련성 임계값 (0.0 ~ 1.0)
            min_image_size: 최소 이미지 크기 (픽셀)
            max_image_size: 최대 이미지 크기 (픽셀)
            enable_ocr: OCR 사용 여부
            use_local_models: 로컬 모델 사용 여부
        """
        self.output_dir = Path(output_dir)
        self.relevance_threshold = relevance_threshold
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.enable_ocr = enable_ocr
        self.use_local_models = use_local_models
        
        # 디렉토리 생성
        self.images_dir = self.output_dir / "images"
        self.texts_dir = self.output_dir / "extracted_texts"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.output_dir, self.images_dir, self.texts_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 모델 초기화
        self.korean_model = None
        self.multimodal_model = None
        if use_local_models:
            self._initialize_models()
        
        # 통계 초기화
        self.reset_statistics()
    
    def _initialize_models(self):
        """로컬 모델 초기화"""
        try:
            logger.info("🚀 로컬 모델 초기화 시작...")
            
            # 한국어 텍스트 모델
            logger.info("   1/2 한국어 텍스트 모델 로딩...")
            self.korean_model = KoreanTextModel(
                n_ctx=4096,
                n_threads=4
            )
            
            # 멀티모달 모델 (A.X 4.0 VL Light)
            logger.info("   2/2 멀티모달 모델 로딩 (A.X 4.0 VL Light)...")
            from src.utils.model_bootstrap import get_ax_vl_dir
            from src.utils.ax_multimodal import AXMultimodalModel
            ax_dir = get_ax_vl_dir()
            if ax_dir.exists():
                self.multimodal_model = AXMultimodalModel(
                    model_path=str(ax_dir),
                    device="auto",
                    max_memory_gb=8,
                )
            else:
                logger.warning("A.X 4.0 VL Light 디렉토리를 찾을 수 없어 이미지 분석을 비활성화합니다.")
                self.multimodal_model = None
            
            logger.info("✅ 모든 모델 초기화 완료!")
            
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")
            logger.info("💡 scripts/download_models.py를 실행하여 모델을 다운로드하세요.")
            self.use_local_models = False
    
    def reset_statistics(self):
        """통계 초기화"""
        self.stats = {
            'total_pages': 0,
            'total_images_found': 0,
            'relevant_images_saved': 0,
            'irrelevant_images_skipped': 0,
            'text_images_converted': 0,
            'ocr_text_extracted': 0,
            'small_images_skipped': 0,
            'errors': 0,
            'processing_time': 0.0
        }
    
    def extract_document_topic(self, pdf_path: str, max_pages: int = 5) -> Dict[str, Any]:
        """
        PDF 문서의 주제 추출
        
        Args:
            pdf_path: PDF 파일 경로
            max_pages: 분석할 최대 페이지 수
            
        Returns:
            문서 주제 정보
        """
        if not self.korean_model:
            logger.warning("한국어 모델이 없어 기본 주제 추출을 수행합니다.")
            return self._extract_basic_topic(pdf_path, max_pages)
        
        try:
            doc = fitz.open(pdf_path)
            
            # 처음 N페이지의 텍스트 추출
            text_content = []
            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_content.append(text.strip())
            
            doc.close()
            
            # 텍스트 결합
            combined_text = "\n".join(text_content)[:3000]  # 최대 3000자
            
            if not combined_text:
                logger.warning("문서에서 텍스트를 추출할 수 없습니다.")
                return {"main_topic": Path(pdf_path).stem, "keywords": []}
            
            # 한국어 모델로 주제 추출
            topic_info = self.korean_model.extract_document_topic(combined_text)

            # 방어 로직: topic_info가 문자열(JSON)인 경우 딕셔너리로 변환 시도
            if isinstance(topic_info, str):
                try:
                    topic_info = json.loads(topic_info)
                except Exception:
                    # 문자열인 경우 최소 구조로 래핑
                    topic_info = {
                        "main_topic": Path(pdf_path).stem,
                        "keywords": [],
                        "raw_response": topic_info
                    }

            # 키워드 방어 로직: 비어있으면 파일명 기반 기본 키워드로 보강
            keywords = topic_info.get('keywords', []) if isinstance(topic_info, dict) else []
            if not keywords:
                basic = self._extract_basic_topic(pdf_path, max_pages)
                keywords = basic.get('keywords', [])
                if isinstance(topic_info, dict):
                    topic_info['keywords'] = keywords

            # 로깅용 키워드 정제 (문자열만, 공백 제거, 상위 5개)
            safe_keywords = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]
            keywords_for_log = ", ".join(safe_keywords[:5]) if safe_keywords else "없음"

            logger.info(f"📚 문서 주제: {topic_info.get('main_topic', 'Unknown')}")
            logger.info(f"   키워드: {keywords_for_log}")

            return topic_info
            
        except Exception as e:
            logger.error(f"주제 추출 실패: {e}")
            return {"main_topic": Path(pdf_path).stem, "keywords": [], "error": str(e)}
    
    def _extract_basic_topic(self, pdf_path: str, max_pages: int = 5) -> Dict[str, Any]:
        """기본 주제 추출 (모델 없이)"""
        # macOS 파일명 한글 정규화(NFD) 문제를 방지하기 위해 NFC로 통일
        pdf_name_raw = Path(pdf_path).stem
        pdf_name = unicodedata.normalize('NFC', pdf_name_raw)
        
        # 파일명에서 키워드 추출
        keywords = []
        if "몽촌" in pdf_name:
            keywords.extend(["몽촌토성", "백제", "토성"])
        if "발굴" in pdf_name:
            keywords.extend(["발굴", "유적", "유물"])
        
        return {
            "main_topic": pdf_name,
            "keywords": keywords,
            "domain": "문서",
            "summary": f"{pdf_name} 관련 문서"
        }
    
    def process_pdf(self, pdf_path: str, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        PDF 파일 처리 - 지능형 이미지 추출
        
        Args:
            pdf_path: PDF 파일 경로
            progress_callback: 진행 상황 콜백
            
        Returns:
            처리 결과
        """
        import time
        start_time = time.time()
        
        self.reset_statistics()
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return self.stats
        
        logger.info(f"📄 PDF 처리 시작: {pdf_path.name}")
        
        try:
            # 1단계: 문서 주제 추출
            if progress_callback:
                progress_callback(0.1, "문서 주제 분석 중...")
            
            document_topic = self.extract_document_topic(str(pdf_path))
            
            # 2단계: PDF 열기 및 이미지 추출
            if progress_callback:
                progress_callback(0.2, "이미지 추출 중...")
            
            doc = fitz.open(str(pdf_path))
            self.stats['total_pages'] = len(doc)
            
            # 결과 저장용
            extraction_results = {
                'pdf_file': str(pdf_path),
                'document_topic': document_topic,
                'images': [],
                'statistics': {}
            }
            
            # 페이지별 처리
            for page_num in range(len(doc)):
                if progress_callback:
                    progress = 0.2 + (page_num / len(doc)) * 0.7
                    progress_callback(progress, f"페이지 {page_num + 1}/{len(doc)} 처리 중...")
                
                page = doc[page_num]
                
                # 페이지 텍스트 (컨텍스트용)
                page_text = page.get_text()[:500]  # 처음 500자
                
                # 이미지 추출
                image_list = page.get_images(full=True)
                self.stats['total_images_found'] += len(image_list)
                
                for img_index, img in enumerate(image_list):
                    result = self._process_single_image(
                        doc, pdf_path, page_num + 1, img_index, img,
                        document_topic, page_text
                    )
                    
                    if result:
                        extraction_results['images'].append(result)
            
            doc.close()
            
            # 3단계: 결과 저장
            if progress_callback:
                progress_callback(0.9, "결과 저장 중...")
            
            self.stats['processing_time'] = time.time() - start_time
            extraction_results['statistics'] = self.stats.copy()
            
            # 보고서 저장
            self._save_extraction_report(pdf_path, extraction_results)
            
            # 최종 통계 출력
            self._print_statistics()
            
            if progress_callback:
                progress_callback(1.0, "처리 완료!")
            
            return extraction_results
            
        except Exception as e:
            logger.error(f"PDF 처리 실패: {e}")
            self.stats['errors'] += 1
            self.stats['processing_time'] = time.time() - start_time
            return self.stats
    
    def _process_single_image(self, doc, pdf_path: Path, page_num: int, img_index: int,
                             img, document_topic: Dict, page_text: str) -> Optional[Dict]:
        """단일 이미지 처리"""
        try:
            # 이미지 추출
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            
            # 크기 확인
            if width < self.min_image_size or height < self.min_image_size:
                self.stats['small_images_skipped'] += 1
                logger.debug(f"   작은 이미지 건너뜀: {width}x{height}")
                return None
            
            # 임시 저장
            temp_filename = f"temp_{pdf_path.stem}_p{page_num:03d}_i{img_index+1:03d}.{image_ext}"
            temp_path = self.output_dir / temp_filename
            
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            
            # 이미지 분석 (멀티모달 모델 사용)
            if self.multimodal_model and self.multimodal_model.model is not None:
                analysis = self.multimodal_model.analyze_image(
                    str(temp_path),
                    document_topic,
                    page_text
                )
                
                relevance_score = analysis.get('relevance_score', 0.5)
                is_text_only = analysis.get('is_text_only', False)
                image_type = analysis.get('image_type', 'unknown')
                description = analysis.get('content_description', '')
                
            else:
                # 모델 없이 기본 처리 - 모든 이미지를 관련있다고 가정
                logger.debug("모델이 없어 기본값으로 이미지 처리")
                relevance_score = 0.8  # 기본값을 높게 설정하여 저장되도록 함
                is_text_only = False
                image_type = 'document_image'
                description = f'페이지 {page_num}의 이미지'
            
            # 처리 결정
            result = {
                'page': page_num,
                'index': img_index + 1,
                'size': f"{width}x{height}",
                'type': image_type,
                'relevance_score': relevance_score,
                'is_text_only': is_text_only,
                'description': description
            }
            
            # 텍스트 전용 이미지 처리
            if is_text_only and self.enable_ocr:
                ocr_text = self._extract_text_from_image(temp_path)
                if ocr_text:
                    # 텍스트 저장
                    text_filename = f"{pdf_path.stem}_p{page_num:03d}_i{img_index+1:03d}.txt"
                    text_path = self.texts_dir / text_filename
                    
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(ocr_text)
                    
                    result['extracted_text'] = ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text
                    result['text_file'] = str(text_path)
                    
                    self.stats['text_images_converted'] += 1
                    self.stats['ocr_text_extracted'] += len(ocr_text)
                    
                    logger.info(f"   📝 텍스트 추출: 페이지 {page_num} 이미지 {img_index+1} ({len(ocr_text)}자)")
                
                # 임시 파일 삭제
                temp_path.unlink()
                
            # 관련 이미지 저장
            elif relevance_score >= self.relevance_threshold:
                # 최종 위치로 이동
                # macOS NFD로 인해 분리되는 한글 파일명을 NFC로 정규화하고 위험 문자를 제거
                safe_stem = TextProcessor.sanitize_filename(pdf_path.stem)
                final_filename = f"{safe_stem}_p{page_num:03d}_i{img_index+1:03d}.{image_ext}"
                final_path = self.images_dir / final_filename
                
                shutil.move(str(temp_path), str(final_path))
                
                result['image_file'] = str(final_path)
                result['saved'] = True
                
                self.stats['relevant_images_saved'] += 1
                
                logger.info(f"   ✅ 이미지 저장: 페이지 {page_num} 이미지 {img_index+1} "
                          f"(관련도: {relevance_score:.2f}, 유형: {image_type})")
            
            # 관련 없는 이미지
            else:
                self.stats['irrelevant_images_skipped'] += 1
                temp_path.unlink()  # 임시 파일 삭제
                result['saved'] = False
                
                logger.debug(f"   ⏭️ 이미지 건너뜀: 페이지 {page_num} 이미지 {img_index+1} "
                           f"(관련도: {relevance_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"이미지 처리 실패 (페이지 {page_num}, 이미지 {img_index+1}): {e}")
            self.stats['errors'] += 1
            return None
    
    def _extract_text_from_image(self, image_path: Path) -> str:
        """이미지에서 텍스트 추출 (OCR)"""
        try:
            # 한글 OCR 설정
            ocr_config = '--psm 3 -l kor+eng'
            
            # OCR 실행
            text = pytesseract.image_to_string(
                str(image_path),
                config=ocr_config
            )
            
            # 텍스트 정리
            text = text.strip()
            
            # 의미있는 텍스트인지 확인
            if len(text) < 10:
                return ""
            
            return text
            
        except Exception as e:
            logger.warning(f"OCR 실패: {e}")
            return ""
    
    def _save_extraction_report(self, pdf_path: Path, results: Dict):
        """추출 결과 보고서 저장"""
        try:
            report_filename = f"{pdf_path.stem}_extraction_report.json"
            report_path = self.reports_dir / report_filename
            
            # JSON으로 저장
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"📊 보고서 저장: {report_path}")
            
            # 요약 보고서도 생성
            summary_filename = f"{pdf_path.stem}_summary.txt"
            summary_path = self.reports_dir / summary_filename
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"PDF 지능형 이미지 추출 보고서\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"파일: {pdf_path.name}\n")
                f.write(f"처리 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"문서 주제: {results['document_topic'].get('main_topic', 'Unknown')}\n")
                f.write(f"키워드: {', '.join(results['document_topic'].get('keywords', []))}\n\n")
                
                f.write(f"통계:\n")
                f.write(f"  - 총 페이지: {self.stats['total_pages']}\n")
                f.write(f"  - 발견된 이미지: {self.stats['total_images_found']}\n")
                f.write(f"  - 저장된 관련 이미지: {self.stats['relevant_images_saved']}\n")
                f.write(f"  - 건너뛴 무관 이미지: {self.stats['irrelevant_images_skipped']}\n")
                f.write(f"  - 텍스트로 변환: {self.stats['text_images_converted']}\n")
                f.write(f"  - 크기 미달로 건너뜀: {self.stats['small_images_skipped']}\n")
                f.write(f"  - 처리 시간: {self.stats['processing_time']:.2f}초\n")
                
                if self.stats['relevant_images_saved'] > 0:
                    f.write(f"\n저장된 이미지:\n")
                    for img in results['images']:
                        if img.get('saved'):
                            f.write(f"  - 페이지 {img['page']}, 이미지 {img['index']}: "
                                  f"{img['type']} (관련도 {img['relevance_score']:.2f})\n")
                            if img.get('description'):
                                f.write(f"    설명: {img['description'][:100]}\n")
            
            logger.info(f"📝 요약 보고서 저장: {summary_path}")
            
        except Exception as e:
            logger.error(f"보고서 저장 실패: {e}")
    
    def _print_statistics(self):
        """통계 출력"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 처리 통계:")
        logger.info(f"   총 페이지: {self.stats['total_pages']}")
        logger.info(f"   발견된 이미지: {self.stats['total_images_found']}")
        logger.info(f"   저장된 관련 이미지: {self.stats['relevant_images_saved']}")
        logger.info(f"   건너뛴 무관 이미지: {self.stats['irrelevant_images_skipped']}")
        logger.info(f"   텍스트로 변환: {self.stats['text_images_converted']}")
        logger.info(f"   추출된 텍스트: {self.stats['ocr_text_extracted']}자")
        logger.info(f"   크기 미달로 건너뜀: {self.stats['small_images_skipped']}")
        logger.info(f"   오류: {self.stats['errors']}")
        logger.info(f"   처리 시간: {self.stats['processing_time']:.2f}초")
        logger.info("=" * 60)
    
    def cleanup(self):
        """리소스 정리"""
        if self.korean_model:
            self.korean_model.cleanup()
        
        if self.multimodal_model:
            self.multimodal_model.cleanup()
        
        logger.info("리소스 정리 완료")


# 편의 함수
def extract_images_intelligently(pdf_path: str, **kwargs) -> Dict[str, Any]:
    """PDF에서 지능적으로 이미지 추출"""
    extractor = IntelligentImageExtractorKorean(**kwargs)
    
    try:
        results = extractor.process_pdf(pdf_path)
        return results
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    # 테스트
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        # 기본 테스트 파일
        pdf_file = "test.pdf"
    
    if Path(pdf_file).exists():
        logger.info(f"테스트 시작: {pdf_file}")
        
        results = extract_images_intelligently(
            pdf_file,
            relevance_threshold=0.6,
            use_local_models=True
        )
        
        logger.info("테스트 완료!")
    else:
        logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_file}")