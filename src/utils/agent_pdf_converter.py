"""
에이전트 기반 PDF 변환기 - 로컬 LLM을 활용한 지능형 문서 전처리
"""

import os
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.agents import ContextConnectorAgent, StructureParserAgent, QualityValidatorAgent
from .pdf_converter import ImprovedPDFConverter
from .image_analyzer import create_image_analyzer

logger = logging.getLogger(__name__)


class AgentBasedPDFConverter:
    """
    로컬 LLM 에이전트들을 활용한 고품질 PDF-to-Markdown 변환기
    """
    
    def __init__(self, output_dir: str = "converted_docs_agent", enable_quality_validation: bool = True, enable_image_analysis: bool = True):
        self.output_dir = output_dir
        self.enable_quality_validation = enable_quality_validation
        self.enable_image_analysis = enable_image_analysis
        
        # 출력 디렉토리 생성
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 에이전트 초기화
        self.context_agent = ContextConnectorAgent()
        self.structure_agent = StructureParserAgent()
        
        if enable_quality_validation:
            self.quality_agent = QualityValidatorAgent()
        
        # 이미지 분석기 초기화 (선택적)
        if enable_image_analysis:
            try:
                self.image_analyzer = create_image_analyzer()
                logger.info("🖼️ 이미지 분석기 활성화됨")
            except Exception as e:
                logger.warning(f"⚠️ 이미지 분석기 초기화 실패: {str(e)}")
                self.image_analyzer = None
                self.enable_image_analysis = False

            # 로컬 멀티모달(Gemma) 직접 연결 옵션: 이미지 추출에서 사용하는 것과 동일한 모델 재사용
            self.local_multimodal = None
            try:
                from src.utils.model_bootstrap import get_ax_vl_dir
                from src.utils.ax_multimodal import AXMultimodalModel
                ax_path = get_ax_vl_dir()
                if ax_path and ax_path.exists():
                    # 동일 경로로 멀티모달 인스턴스 구성 (A.X-4.0-VL-Light)
                    self.local_multimodal = AXMultimodalModel(model_path=str(ax_path), device="auto", max_memory_gb=8)
                    logger.info("🔗 로컬 Gemma 멀티모달을 이미지 설명에도 재사용합니다")
            except Exception as e:
                logger.warning(f"로컬 멀티모달 연결 건너뜀: {e}")
        else:
            self.image_analyzer = None
        
        # 기존 변환기 (비교용)
        self.fallback_converter = ImprovedPDFConverter(output_dir=f"{output_dir}_fallback")
        
        logger.info("🤖 에이전트 기반 PDF 변환기 초기화 완료")
    
    def convert_pdf_to_markdown(self, pdf_path: str, comparison_mode: bool = False) -> str:
        """
        PDF를 고품질 마크다운으로 변환
        
        Args:
            pdf_path: 변환할 PDF 파일 경로
            comparison_mode: True면 기존 방식과 비교 결과도 생성
            
        Returns:
            변환된 마크다운 파일 경로
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        logger.info(f"🚀 에이전트 기반 PDF 변환 시작: {pdf_path}")
        
        # 1단계: 기본 텍스트 추출
        text_blocks = self._extract_text_blocks(pdf_path)
        logger.info(f"📄 텍스트 블록 추출 완료: {len(text_blocks)}개 블록")
        
        # 2단계: 문맥 연결 에이전트 적용
        connected_blocks = self.context_agent.process(text_blocks)
        logger.info(f"🔗 문맥 연결 완료: {len(text_blocks)} → {len(connected_blocks)}개 블록")
        
        # 3단계: 구조 파싱 에이전트 적용
        full_content = '\n\n'.join(connected_blocks)
        structured_content = self.structure_agent.process(full_content)
        logger.info("🏗️ 문서 구조화 완료")
        
        # 4단계: 품질 검증 (선택적)
        quality_report = None
        if self.enable_quality_validation:
            quality_report = self.quality_agent.process(structured_content)
            logger.info(f"✅ 품질 검증 완료 - 점수: {quality_report['overall_score']:.2f}")
            
            # 품질이 낮으면 개선 제안 로깅
            if quality_report['overall_score'] < 0.8:
                logger.warning("⚠️ 품질 개선이 필요합니다:")
                for improvement in quality_report['improvements']:
                    logger.warning(f"   💡 {improvement}")
        
        # 5단계: 결과 저장
        output_path = self._save_result(pdf_path, structured_content, quality_report)
        
        # 6단계: 비교 모드 실행 (선택적)
        if comparison_mode:
            self._run_comparison(pdf_path, structured_content, quality_report)
        
        logger.info(f"🎉 에이전트 기반 변환 완료: {output_path}")
        return output_path
    
    def _extract_text_blocks(self, pdf_path: str) -> List[str]:
        """
        PDF에서 페이지별 텍스트 블록 추출 및 이미지 추출
        에이전트 모드용: 문맥 연결된 텍스트 제공
        """
        try:
            doc = fitz.open(pdf_path)
            pdf_stem = Path(pdf_path).stem
            
            # 이미지 디렉토리 생성
            images_dir = Path(self.output_dir) / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # 이미지 정보 저장을 위한 딕셔너리
            extracted_images = {}  # {page_num: [(image_path, description), ...]}
            
            # 1단계: 모든 페이지 텍스트 수집 및 이미지 추출
            page_texts = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():
                    page_texts.append((page_num + 1, page_text.strip()))
                
                # 페이지에서 이미지 추출
                try:
                    image_list = page.get_images(full=True)
                    page_images = []
                    MIN_IMAGE_SIZE = 50  # 최소 이미지 크기 (픽셀)
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]  # xref 번호
                        
                        # 이미지 데이터 추출
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # 이미지 크기 확인을 위해 PIL 사용
                        from PIL import Image
                        import io
                        img_pil = Image.open(io.BytesIO(image_bytes))
                        width, height = img_pil.size
                        
                        # 너무 작은 이미지는 건너뛰기
                        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                            logger.info(f"   ⚠️  너무 작은 이미지 건너뛰기: {width}x{height} (페이지 {page_num + 1})")
                            continue
                        
                        # 이미지 파일명 생성
                        image_filename = f"{pdf_stem}_page{page_num + 1:03d}_img{img_index + 1:03d}.{image_ext}"
                        image_path = images_dir / image_filename
                        
                        # 이미지 저장
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        logger.info(f"   🖼️  이미지 추출 성공: {image_filename} ({width}x{height})")
                        
                        # 이미지 분석 및 설명 생성
                        image_description = None
                        if self.enable_image_analysis:
                            # 우선 로컬 멀티모달을 사용하고, 실패 시 image_analyzer로 폴백
                            context = page_text.strip()[:500] if page_text.strip() else ""
                            desc_ok = False
                            if getattr(self, 'local_multimodal', None):
                                try:
                                    analysis = self.local_multimodal.analyze_image(str(image_path), {"main_topic": "", "keywords": []}, context)
                                    image_description = (analysis or {}).get("content_description")
                                    if image_description:
                                        logger.info(f"   ✅ 이미지 설명 생성: {len(image_description)}자")
                                        desc_ok = True
                                except Exception as e:
                                    logger.warning(f"   ⚠️ 로컬 멀티모달 설명 실패: {e}")
                            if not desc_ok and self.image_analyzer:
                                try:
                                    image_description = self.image_analyzer.analyze_image(str(image_path), context)
                                    if image_description:
                                        logger.info(f"   ✅ 이미지 설명 생성: {len(image_description)}자")
                                    else:
                                        logger.warning(f"   ⚠️ 이미지 설명 생성 실패: {image_filename}")
                                except Exception as e:
                                    logger.warning(f"   ⚠️ 이미지 분석 오류: {str(e)}")
                        
                        # 상대 경로로 저장 (마크다운에서 사용)
                        relative_image_path = f"./images/{image_filename}"
                        page_images.append((relative_image_path, image_description))
                    
                    if page_images:
                        extracted_images[page_num + 1] = page_images
                        
                except Exception as e:
                    logger.warning(f"페이지 {page_num + 1} 이미지 추출 실패: {str(e)}")
                    
            doc.close()
            
            # 2단계: 추출된 이미지 개수 로깅
            total_images = sum(len(imgs) for imgs in extracted_images.values())
            if total_images > 0:
                logger.info(f"📊 총 {total_images}개 이미지 추출 및 분석 완료")
            
            # 3단계: page_texts를 text_blocks로 변환 (이미지 정보 포함)
            text_blocks = []
            for page_num, text in page_texts:
                # 페이지 헤더와 텍스트 추가
                block_content = f"[페이지 {page_num}]\n{text}"
                
                # 해당 페이지의 이미지가 있으면 추가
                if page_num in extracted_images:
                    block_content += "\n\n### 페이지 내 이미지\n"
                    
                    for img_path, img_description in extracted_images[page_num]:
                        block_content += f"\n![이미지]({img_path})\n"
                        
                        if img_description:
                            block_content += f"**이미지 설명**: {img_description}\n"
                        else:
                            block_content += f"**이미지**: 페이지 {page_num}의 이미지 {img_path}\n"
                
                text_blocks.append(block_content)
            
            if not text_blocks:
                logger.warning("⚠️ PDF에서 텍스트를 추출할 수 없습니다.")
                text_blocks = [""]  # 빈 리스트 대신 빈 문자열 하나를 포함
            
        except Exception as e:
            logger.error(f"❌ PDF 텍스트/이미지 추출 실패: {str(e)}")
            raise
        
        return text_blocks
    
    def _save_result(self, pdf_path: str, content: str, quality_report: Optional[Dict] = None) -> str:
        """
        변환 결과 저장
        """
        # 출력 파일명 생성
        pdf_name = Path(pdf_path).stem
        md_filename = f"{pdf_name}.md"
        output_path = os.path.join(self.output_dir, md_filename)
        
        # 이미지 개수 계산
        images_dir = Path(self.output_dir) / "images"
        image_count = 0
        if images_dir.exists():
            image_count = len([f for f in images_dir.glob(f"{pdf_name}_page*_img*.*")])
        
        # 메타데이터 헤더 생성
        from datetime import datetime
        metadata_header = f"""# {pdf_name}

**원본 파일**: {Path(pdf_path).name}
**변환 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**처리 방법**: 에이전트 기반 변환기
**추출된 이미지**: {image_count}개

"""
        
        # 마크다운 파일 저장 (메타데이터 포함)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(metadata_header + content)
        
        # 품질 보고서 저장 (있는 경우)
        if quality_report:
            quality_path = os.path.join(self.output_dir, f"{pdf_name}_quality_report.json")
            import json
            with open(quality_path, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, ensure_ascii=False, indent=2)
            logger.info(f"📊 품질 보고서 저장: {quality_path}")
        
        return output_path
    
    def _run_comparison(self, pdf_path: str, agent_content: str, quality_report: Optional[Dict] = None):
        """
        기존 방식과 품질 비교
        """
        logger.info("🔄 기존 방식과 비교 분석 시작...")
        
        try:
            # 기존 방식으로 변환
            fallback_path = self.fallback_converter.convert_pdf_to_markdown(pdf_path)
            
            with open(fallback_path, 'r', encoding='utf-8') as f:
                fallback_content = f.read()
            
            # 간단한 비교 메트릭
            comparison = {
                'agent_length': len(agent_content),
                'fallback_length': len(fallback_content),
                'agent_lines': len(agent_content.split('\n')),
                'fallback_lines': len(fallback_content.split('\n')),
                'agent_quality_score': quality_report['overall_score'] if quality_report else 0.0,
                'improvement_percentage': 0.0
            }
            
            if quality_report:
                # 기존 방식 품질도 검증
                fallback_quality = self.quality_agent.process(fallback_content)
                comparison['fallback_quality_score'] = fallback_quality['overall_score']
                comparison['improvement_percentage'] = (
                    (comparison['agent_quality_score'] - comparison['fallback_quality_score']) 
                    / max(comparison['fallback_quality_score'], 0.1) * 100
                )
            
            # 비교 결과 저장
            pdf_name = Path(pdf_path).stem
            comparison_path = os.path.join(self.output_dir, f"{pdf_name}_comparison.json")
            
            import json
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📈 비교 분석 완료:")
            logger.info(f"   에이전트 방식 품질: {comparison['agent_quality_score']:.2f}")
            if 'fallback_quality_score' in comparison:
                logger.info(f"   기존 방식 품질: {comparison['fallback_quality_score']:.2f}")
                logger.info(f"   개선율: {comparison['improvement_percentage']:.1f}%")
            
        except Exception as e:
            logger.warning(f"⚠️ 비교 분석 실패: {str(e)}")
    
    def batch_convert(self, pdf_files: List[str], comparison_mode: bool = False) -> List[str]:
        """
        여러 PDF 파일을 배치로 변환
        """
        results = []
        
        logger.info(f"📦 배치 변환 시작: {len(pdf_files)}개 파일")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                logger.info(f"🔄 진행 상황: {i}/{len(pdf_files)} - {os.path.basename(pdf_path)}")
                result_path = self.convert_pdf_to_markdown(pdf_path, comparison_mode)
                results.append(result_path)
            except Exception as e:
                logger.error(f"❌ 변환 실패: {pdf_path} - {str(e)}")
                results.append(None)
        
        successful_count = sum(1 for r in results if r is not None)
        logger.info(f"📦 배치 변환 완료: {successful_count}/{len(pdf_files)}개 성공")
        
        return results