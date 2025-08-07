"""
지능형 PDF 변환기 - 이미지 분류를 통한 스마트 문서 변환

이 모듈은 기존 PDF 변환기를 확장하여 다음 기능을 제공합니다:
- 텍스트 전용 스캔 이미지는 OCR로 텍스트 추출하여 MD에 포함
- 실제 사진/그림/다이어그램만 이미지로 보존
- 분류 결과 및 통계 제공
"""

import os
import logging
from typing import Tuple, Dict, List, Optional, Callable
from pathlib import Path
import fitz  # PyMuPDF
from datetime import datetime

from .pdf_converter import ImprovedPDFConverter
from .image_classifier import ImageClassifier
from .agent_pdf_converter import AgentBasedPDFConverter

logger = logging.getLogger(__name__)


class IntelligentPDFConverter:
    """이미지 분류 기능이 통합된 지능형 PDF 변환기"""
    
    def __init__(self, 
                 output_dir: str = "converted_docs_intelligent",
                 enable_semantic_chunking: bool = True,
                 enable_postprocessing: bool = False,
                 enable_image_classification: bool = True,
                 classification_config: Optional[Dict] = None):
        """
        Args:
            output_dir: 출력 디렉토리
            enable_semantic_chunking: 의미 기반 청킹 사용 여부
            enable_postprocessing: 후처리 사용 여부  
            enable_image_classification: 이미지 분류 사용 여부
            classification_config: 이미지 분류기 설정
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images" 
        self.texts_dir = self.output_dir / "extracted_texts"
        self.reports_dir = self.output_dir / "classification_reports"
        
        # 출력 디렉토리 생성
        for dir_path in [self.output_dir, self.images_dir, self.texts_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 기본 PDF 변환기 초기화
        self.base_converter = ImprovedPDFConverter(
            output_dir=str(self.output_dir),
            enable_semantic_chunking=enable_semantic_chunking,
            enable_postprocessing=enable_postprocessing
        )
        
        # 이미지 분류기 초기화
        self.enable_image_classification = enable_image_classification
        if enable_image_classification:
            try:
                config = classification_config or {}
                self.image_classifier = ImageClassifier(**config)
                logger.info("🧠 이미지 분류기 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ 이미지 분류기 초기화 실패: {str(e)}")
                self.enable_image_classification = False
                self.image_classifier = None
        else:
            self.image_classifier = None
            
        # 통계 초기화
        self.reset_statistics()
    
    def reset_statistics(self):
        """통계 초기화"""
        self.stats = {
            'total_images_found': 0,
            'text_images_converted': 0,
            'actual_images_preserved': 0,
            'classification_errors': 0,
            'total_extracted_text_length': 0,
            'processing_time': 0.0
        }
    
    def convert_pdf_to_markdown(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[Optional[str], Dict]:
        """
        PDF를 마크다운으로 변환 (이미지 분류 포함)
        
        Args:
            pdf_path: PDF 파일 경로
            progress_callback: 진행 상황 콜백 함수
        
        Returns:
            Tuple[마크다운 내용, 통계 및 결과 정보]
        """
        import time
        start_time = time.time()
        
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
                return None, self.stats
            
            self.reset_statistics()
            
            if progress_callback:
                progress_callback(0.1, "PDF 분석 시작...")
            
            # 1단계: PDF에서 텍스트와 이미지 정보 추출
            doc = fitz.open(pdf_path)
            page_texts, page_images = self._extract_pdf_content(doc, pdf_path, progress_callback)
            
            # 2단계: 이미지 분류 및 처리 (활성화된 경우)
            if self.enable_image_classification:
                if progress_callback:
                    progress_callback(0.4, "이미지 분류 및 처리...")
                classified_images = self._classify_and_process_images(doc, pdf_path, page_images, progress_callback)
            else:
                if progress_callback:
                    progress_callback(0.4, "전통적 이미지 추출...")
                classified_images = self._traditional_image_processing(doc, pdf_path, page_images, progress_callback)
            
            doc.close()
            
            # 3단계: 텍스트 연결 및 마크다운 생성
            if progress_callback:
                progress_callback(0.7, "텍스트 연결 및 마크다운 생성...")
            
            markdown_content = self._generate_enhanced_markdown(
                pdf_path, page_texts, classified_images, progress_callback
            )
            
            # 4단계: 결과 저장
            if progress_callback:
                progress_callback(0.9, "결과 저장 중...")
            
            md_filename = f"{pdf_path.stem}.md"
            md_path = self.output_dir / md_filename
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # 5단계: 분류 보고서 저장
            if self.enable_image_classification:
                self._save_classification_report(pdf_path, classified_images)
            
            # 통계 업데이트
            self.stats['processing_time'] = time.time() - start_time
            
            if progress_callback:
                progress_callback(1.0, f"변환 완료: {self.stats['actual_images_preserved']}개 이미지, "
                                      f"{self.stats['text_images_converted']}개 텍스트 변환")
            
            logger.info(f"지능형 PDF 변환 완료: {pdf_path.name} -> {md_path}")
            logger.info(f"  📊 통계: 총 {self.stats['total_images_found']}개 이미지 중 "
                       f"{self.stats['actual_images_preserved']}개 보존, {self.stats['text_images_converted']}개 텍스트 변환")
            
            return markdown_content, self.stats
            
        except Exception as e:
            self.stats['processing_time'] = time.time() - start_time
            logger.error(f"지능형 PDF 변환 실패: {pdf_path} - {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"변환 실패: {str(e)}")
            return None, self.stats
    
    def _extract_pdf_content(self, doc, pdf_path: Path, progress_callback: Optional[Callable]) -> Tuple[List, List]:
        """PDF에서 텍스트와 이미지 정보 추출"""
        page_texts = []
        page_images = []
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            if progress_callback:
                progress = 0.1 + (page_num / total_pages) * 0.2
                progress_callback(progress, f"페이지 {page_num + 1}/{total_pages} 분석 중...")
            
            page = doc[page_num]
            
            # 텍스트 추출
            text = page.get_text()
            page_texts.append((page_num + 1, text.strip() if text else ""))
            
            # 이미지 정보 추출
            image_list = page.get_images(full=True)
            page_images.append((page_num + 1, image_list))
            
            self.stats['total_images_found'] += len(image_list)
        
        return page_texts, page_images
    
    def _classify_and_process_images(self, doc, pdf_path: Path, page_images: List, progress_callback: Optional[Callable]) -> Dict:
        """이미지 분류 및 지능적 처리"""
        classified_images = {
            'actual_images': [],      # 실제 이미지 (보존)
            'text_images': [],        # 텍스트 이미지 (텍스트로 변환)
            'classification_results': {},  # 상세 분류 결과
            'extracted_texts': {}     # 텍스트 이미지에서 추출한 텍스트
        }
        
        total_images = sum(len(images) for _, images in page_images)
        processed_images = 0
        
        for page_num, image_list in page_images:
            for img_index, img in enumerate(image_list):
                try:
                    if progress_callback and total_images > 0:
                        progress = 0.4 + (processed_images / total_images) * 0.25
                        progress_callback(progress, f"이미지 {processed_images + 1}/{total_images} 분류 중...")
                    
                    # 이미지 추출 및 임시 저장
                    temp_image_path = self._extract_single_image(doc, pdf_path, page_num, img_index, img)
                    
                    if temp_image_path and temp_image_path.exists():
                        # 이미지 분류 수행
                        classification = self.image_classifier.classify_image(str(temp_image_path))
                        
                        image_info = {
                            'page_num': page_num,
                            'img_index': img_index,
                            'temp_path': temp_image_path,
                            'classification': classification
                        }
                        
                        # 분류 결과에 따라 처리
                        if classification['is_text_only'] and classification['confidence'] > 0.6:
                            # 텍스트 이미지로 분류 - 텍스트 추출
                            self._process_text_image(image_info, classified_images)
                            self.stats['text_images_converted'] += 1
                            self.stats['total_extracted_text_length'] += len(classification.get('text_content', ''))
                        else:
                            # 실제 이미지로 분류 - 이미지 보존
                            self._process_actual_image(image_info, classified_images, pdf_path)
                            self.stats['actual_images_preserved'] += 1
                        
                        # 분류 결과 저장
                        image_key = f"page{page_num}_img{img_index+1}"
                        classified_images['classification_results'][image_key] = classification
                        
                    processed_images += 1
                    
                except Exception as e:
                    logger.warning(f"이미지 분류 실패 (페이지 {page_num}, 이미지 {img_index + 1}): {e}")
                    self.stats['classification_errors'] += 1
                    processed_images += 1
        
        return classified_images
    
    def _traditional_image_processing(self, doc, pdf_path: Path, page_images: List, progress_callback: Optional[Callable]) -> Dict:
        """전통적 이미지 처리 (모든 이미지 보존)"""
        classified_images = {
            'actual_images': [],
            'text_images': [],
            'classification_results': {},
            'extracted_texts': {}
        }
        
        total_images = sum(len(images) for _, images in page_images)
        processed_images = 0
        
        for page_num, image_list in page_images:
            for img_index, img in enumerate(image_list):
                try:
                    if progress_callback and total_images > 0:
                        progress = 0.4 + (processed_images / total_images) * 0.25
                        progress_callback(progress, f"이미지 {processed_images + 1}/{total_images} 추출 중...")
                    
                    # 이미지 추출
                    temp_image_path = self._extract_single_image(doc, pdf_path, page_num, img_index, img)
                    
                    if temp_image_path and temp_image_path.exists():
                        # 실제 이미지로 처리 (분류 없이)
                        image_info = {
                            'page_num': page_num,
                            'img_index': img_index,
                            'temp_path': temp_image_path,
                            'classification': {'is_text_only': False, 'confidence': 1.0}
                        }
                        
                        self._process_actual_image(image_info, classified_images, pdf_path)
                        self.stats['actual_images_preserved'] += 1
                    
                    processed_images += 1
                    
                except Exception as e:
                    logger.warning(f"이미지 추출 실패 (페이지 {page_num}, 이미지 {img_index + 1}): {e}")
                    processed_images += 1
        
        return classified_images
    
    def _extract_single_image(self, doc, pdf_path: Path, page_num: int, img_index: int, img) -> Optional[Path]:
        """단일 이미지 추출"""
        try:
            # 이미지 데이터 추출
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # 임시 파일명 생성
            temp_filename = f"temp_{pdf_path.stem}_page{page_num:03d}_img{img_index+1:03d}.{image_ext}"
            temp_path = self.output_dir / temp_filename
            
            # 이미지 저장
            with open(temp_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            return temp_path
            
        except Exception as e:
            logger.warning(f"이미지 추출 실패: {str(e)}")
            return None
    
    def _process_text_image(self, image_info: Dict, classified_images: Dict):
        """텍스트 이미지 처리 (텍스트 추출 및 저장)"""
        classification = image_info['classification']
        text_content = classification.get('text_content', '')
        
        if text_content:
            # 추출된 텍스트 파일로 저장
            page_num = image_info['page_num']
            img_index = image_info['img_index']
            text_filename = f"page{page_num:03d}_img{img_index+1:03d}_extracted_text.txt"
            text_path = self.texts_dir / text_filename
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # 텍스트 이미지 정보 저장
            classified_images['text_images'].append({
                'page_num': page_num,
                'img_index': img_index,
                'text_file': text_path,
                'text_content': text_content,
                'confidence': classification['confidence']
            })
            
            classified_images['extracted_texts'][f"page{page_num}_img{img_index+1}"] = text_content
            
            logger.info(f"  📄 텍스트 추출: 페이지 {page_num} 이미지 {img_index+1} -> {len(text_content)}자")
        
        # 임시 이미지 파일 삭제 (텍스트로 변환되었으므로)
        temp_path = image_info['temp_path']
        if temp_path.exists():
            temp_path.unlink()
    
    def _process_actual_image(self, image_info: Dict, classified_images: Dict, pdf_path: Path):
        """실제 이미지 처리 (이미지 파일로 보존)"""
        temp_path = image_info['temp_path']
        page_num = image_info['page_num']
        img_index = image_info['img_index']
        
        # 최종 이미지 파일명 생성
        image_ext = temp_path.suffix
        final_filename = f"{pdf_path.stem}_page{page_num:03d}_img{img_index+1:03d}{image_ext}"
        final_path = self.images_dir / final_filename
        
        # 임시 파일을 최종 위치로 이동
        temp_path.rename(final_path)
        
        # 이미지 정보 저장
        classified_images['actual_images'].append({
            'page_num': page_num,
            'img_index': img_index,
            'image_file': final_path,
            'relative_path': f"images/{final_filename}",
            'confidence': image_info['classification']['confidence']
        })
        
        logger.info(f"  🖼️ 이미지 보존: 페이지 {page_num} 이미지 {img_index+1} -> {final_filename}")
    
    def _generate_enhanced_markdown(self, pdf_path: Path, page_texts: List, classified_images: Dict, progress_callback: Optional[Callable]) -> str:
        """향상된 마크다운 생성 (분류된 이미지 정보 포함)"""
        markdown_content = []
        
        # 메타데이터 헤더
        markdown_content.append(f"# {pdf_path.stem}")
        markdown_content.append(f"\n**소스 파일:** {pdf_path.name}")
        markdown_content.append(f"**변환 날짜:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append(f"**변환 방식:** 지능형 이미지 분류 {'활성화' if self.enable_image_classification else '비활성화'}")
        
        if self.enable_image_classification:
            markdown_content.append(f"**이미지 통계:** 총 {self.stats['total_images_found']}개 중 "
                                  f"{self.stats['actual_images_preserved']}개 보존, "
                                  f"{self.stats['text_images_converted']}개 텍스트 변환")
        else:
            markdown_content.append(f"**추출된 이미지:** {self.stats['actual_images_preserved']}개")
        
        markdown_content.append("\n---\n")
        
        # 페이지별 텍스트 연결 (기존 로직 활용)
        connected_text = self.base_converter._connect_cross_page_text(page_texts)
        
        # 텍스트 이미지에서 추출한 내용을 적절한 위치에 삽입
        enhanced_text = self._insert_extracted_texts(connected_text, classified_images['extracted_texts'])
        
        # 실제 이미지 참조를 적절한 위치에 삽입
        final_text = self._insert_image_references(enhanced_text, classified_images['actual_images'])
        
        # 텍스트 정리 및 포맷팅
        cleaned_text = self.base_converter._clean_and_format_text(final_text)
        if cleaned_text.strip():
            markdown_content.append(cleaned_text)
        
        # 추가 섹션들
        if classified_images['text_images']:
            markdown_content.append("\n\n## 텍스트로 변환된 이미지\n")
            for text_img in classified_images['text_images']:
                markdown_content.append(f"### 페이지 {text_img['page_num']} 이미지 {text_img['img_index']+1}")
                markdown_content.append(f"*분류 신뢰도: {text_img['confidence']:.2f}*")
                markdown_content.append(f"\n{text_img['text_content']}\n")
        
        if classified_images['actual_images']:
            markdown_content.append("\n\n## 보존된 이미지\n")
            for actual_img in classified_images['actual_images']:
                markdown_content.append(f"![이미지 페이지 {actual_img['page_num']} 이미지 {actual_img['img_index']+1}]({actual_img['relative_path']})")
                markdown_content.append(f"*페이지 {actual_img['page_num']}, 분류 신뢰도: {actual_img['confidence']:.2f}*\n")
        
        return "\n".join(markdown_content)
    
    def _insert_extracted_texts(self, original_text: str, extracted_texts: Dict[str, str]) -> str:
        """추출된 텍스트를 원본 텍스트의 적절한 위치에 삽입"""
        if not extracted_texts:
            return original_text
        
        # 간단한 구현: 각 페이지 끝에 해당 페이지의 추출된 텍스트 추가
        lines = original_text.split('\n')
        enhanced_lines = []
        current_page = 1
        
        for line in lines:
            enhanced_lines.append(line)
            
            # 페이지 전환 감지 (간단한 휴리스틱)
            if f'페이지 {current_page}' in line.lower():
                # 해당 페이지의 추출된 텍스트가 있으면 추가
                page_texts = [text for key, text in extracted_texts.items() 
                            if f'page{current_page}_' in key]
                
                for text in page_texts:
                    enhanced_lines.append(f"\n<!-- 이미지에서 추출된 텍스트 시작 -->")
                    enhanced_lines.append(text)
                    enhanced_lines.append("<!-- 이미지에서 추출된 텍스트 끝 -->\n")
                
                current_page += 1
        
        return '\n'.join(enhanced_lines)
    
    def _insert_image_references(self, text: str, actual_images: List[Dict]) -> str:
        """실제 이미지 참조를 텍스트의 적절한 위치에 삽입"""
        if not actual_images:
            return text
        
        # 페이지별로 이미지 그룹화
        images_by_page = {}
        for img in actual_images:
            page = img['page_num']
            if page not in images_by_page:
                images_by_page[page] = []
            images_by_page[page].append(img)
        
        lines = text.split('\n')
        enhanced_lines = []
        current_page = 1
        
        for line in lines:
            enhanced_lines.append(line)
            
            # 페이지 전환 감지
            if f'페이지 {current_page}' in line.lower():
                # 해당 페이지의 이미지가 있으면 추가
                if current_page in images_by_page:
                    enhanced_lines.append(f"\n<!-- 페이지 {current_page} 이미지들 -->")
                    for img in images_by_page[current_page]:
                        enhanced_lines.append(f"![이미지 {img['img_index']+1}]({img['relative_path']})")
                    enhanced_lines.append("<!-- 이미지 끝 -->\n")
                
                current_page += 1
        
        return '\n'.join(enhanced_lines)
    
    def _save_classification_report(self, pdf_path: Path, classified_images: Dict):
        """분류 보고서 저장"""
        try:
            report = {
                'pdf_file': str(pdf_path),
                'processing_date': datetime.now().isoformat(),
                'statistics': self.stats,
                'classification_results': classified_images['classification_results'],
                'text_images': [
                    {
                        'page': img['page_num'],
                        'index': img['img_index'],
                        'text_length': len(img['text_content']),
                        'confidence': img['confidence']
                    }
                    for img in classified_images['text_images']
                ],
                'actual_images': [
                    {
                        'page': img['page_num'],
                        'index': img['img_index'],
                        'file': str(img['image_file']),
                        'confidence': img['confidence']
                    }
                    for img in classified_images['actual_images']
                ]
            }
            
            report_filename = f"{pdf_path.stem}_classification_report.json"
            report_path = self.reports_dir / report_filename
            
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 분류 보고서 저장: {report_path}")
            
        except Exception as e:
            logger.warning(f"분류 보고서 저장 실패: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """현재 통계 반환"""
        return self.stats.copy()
    
    def batch_convert(self, pdf_files: List[str], progress_callback: Optional[Callable] = None) -> List[Tuple[Optional[str], Dict]]:
        """여러 PDF 파일 일괄 변환"""
        results = []
        total_files = len(pdf_files)
        
        logger.info(f"📦 지능형 배치 변환 시작: {total_files}개 파일")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                if progress_callback:
                    progress_callback((i-1)/total_files, f"파일 {i}/{total_files}: {Path(pdf_path).name}")
                
                logger.info(f"🔄 진행 상황: {i}/{total_files} - {Path(pdf_path).name}")
                result = self.convert_pdf_to_markdown(pdf_path)
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ 변환 실패: {pdf_path} - {str(e)}")
                results.append((None, {'error': str(e)}))
        
        if progress_callback:
            progress_callback(1.0, "배치 변환 완료")
        
        successful_count = sum(1 for r, _ in results if r is not None)
        logger.info(f"📦 배치 변환 완료: {successful_count}/{total_files}개 성공")
        
        return results


# 편의 함수
def convert_pdf_with_intelligent_classification(pdf_path: str, **kwargs) -> Tuple[Optional[str], Dict]:
    """단일 PDF 지능형 변환 편의 함수"""
    converter = IntelligentPDFConverter(**kwargs)
    return converter.convert_pdf_to_markdown(pdf_path)