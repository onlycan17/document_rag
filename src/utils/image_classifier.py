"""
이미지 분류기 - 텍스트 전용 스캔 이미지와 실제 사진/그림 구분

이 모듈은 PDF에서 추출된 이미지를 분석하여 다음을 구분합니다:
- 텍스트만 있는 스캔 이미지 (MD 텍스트로 변환 필요)
- 실제 사진/그림/다이어그램 (이미지로 유지)
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np
import pytesseract
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path
from PIL import Image, ImageStat
import re

logger = logging.getLogger(__name__)


class ImageClassifier:
    """이미지를 텍스트 전용과 실제 이미지로 분류하는 클래스"""
    
    def __init__(self, 
                 text_ratio_threshold: float = 0.7,  # 텍스트 비율 임계값
                 edge_density_threshold: float = 0.3,  # 엣지 밀도 임계값
                 color_variance_threshold: float = 50.0,  # 색상 분산 임계값
                 min_text_length: int = 50,  # 최소 텍스트 길이
                 use_advanced_analysis: bool = True):  # 고급 분석 사용 여부
        """
        Args:
            text_ratio_threshold: 텍스트로 판단할 텍스트 비율 (0.7 = 70%)
            edge_density_threshold: 엣지 밀도 임계값 (높을수록 텍스트 가능성 높음)
            color_variance_threshold: 색상 분산 임계값 (낮을수록 텍스트 가능성 높음)
            min_text_length: OCR로 추출한 텍스트의 최소 길이
        """
        self.text_ratio_threshold = text_ratio_threshold
        self.edge_density_threshold = edge_density_threshold
        self.color_variance_threshold = color_variance_threshold
        self.min_text_length = min_text_length
        self.use_advanced_analysis = use_advanced_analysis and CV2_AVAILABLE
        
        # OpenCV 없는 경우 경고
        if use_advanced_analysis and not CV2_AVAILABLE:
            logger.warning("OpenCV가 설치되지 않아 기본 OCR 분석만 사용합니다. 완전한 기능을 위해 'pip install opencv-python' 실행하세요.")
        
        # Tesseract OCR 설정 (한글 지원)
        self.ocr_config = '--psm 3 -l kor+eng'
        
    def classify_image(self, image_path: str) -> Dict[str, any]:
        """
        이미지를 분류하고 상세 분석 결과 반환
        
        Args:
            image_path: 분석할 이미지 경로
            
        Returns:
            Dict with keys:
            - is_text_only: bool, 텍스트 전용 이미지 여부
            - confidence: float, 판단 신뢰도 (0-1)
            - text_content: str, OCR로 추출한 텍스트 (텍스트 전용인 경우)
            - analysis_details: Dict, 상세 분석 결과
        """
        try:
            if not Path(image_path).exists():
                logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                return self._default_result()
            
            # OCR 분석 (항상 수행)
            ocr_result = self._analyze_ocr_content(image_path)
            
            # OpenCV 기반 고급 분석 (사용 가능하고 활성화된 경우에만)
            if self.use_advanced_analysis and CV2_AVAILABLE:
                # 이미지 로드
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"이미지를 읽을 수 없습니다: {image_path}")
                    return self._default_result()
                
                # 고급 분석 수행
                color_analysis = self._analyze_color_characteristics(image)
                edge_analysis = self._analyze_edge_characteristics(image)
                texture_analysis = self._analyze_texture_complexity(image)
                
                # 종합 분석 및 분류
                classification = self._make_classification_decision(
                    ocr_result, color_analysis, edge_analysis, texture_analysis
                )
                
                # 결과 패키징
                result = {
                    'is_text_only': classification['is_text_only'],
                    'confidence': classification['confidence'],
                    'text_content': ocr_result['text'] if classification['is_text_only'] else '',
                    'analysis_details': {
                        'ocr_analysis': ocr_result,
                        'color_analysis': color_analysis,
                        'edge_analysis': edge_analysis,
                        'texture_analysis': texture_analysis,
                        'decision_factors': classification['factors'],
                        'analysis_mode': 'advanced'
                    }
                }
                
            else:
                # 기본 OCR 기반 분류 (OpenCV 없음)
                classification = self._make_basic_classification_decision(ocr_result)
                
                # 결과 패키징
                result = {
                    'is_text_only': classification['is_text_only'],
                    'confidence': classification['confidence'],
                    'text_content': ocr_result['text'] if classification['is_text_only'] else '',
                    'analysis_details': {
                        'ocr_analysis': ocr_result,
                        'decision_factors': classification['factors'],
                        'analysis_mode': 'basic_ocr_only'
                    }
                }
            
            logger.info(f"이미지 분류 완료: {Path(image_path).name} -> "
                       f"{'텍스트 전용' if result['is_text_only'] else '실제 이미지'} "
                       f"(신뢰도: {result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"이미지 분류 중 오류 발생: {str(e)}")
            return self._default_result()
    
    def _analyze_ocr_content(self, image_path: str) -> Dict[str, any]:
        """OCR을 통한 텍스트 내용 분석"""
        try:
            # PIL로 이미지 열기 (OCR 성능 향상)
            pil_image = Image.open(image_path)
            
            # 그레이스케일 변환 (OCR 성능 향상)
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            
            # OCR 실행
            extracted_text = pytesseract.image_to_string(pil_image, config=self.ocr_config)
            
            # 텍스트 정리
            cleaned_text = self._clean_ocr_text(extracted_text)
            
            # 신뢰도 계산 (OCR 상세 결과 사용)
            ocr_data = pytesseract.image_to_data(pil_image, config=self.ocr_config, output_type=pytesseract.Output.DICT)
            confidence_scores = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            # 텍스트 영역 비율 계산
            text_area_ratio = self._calculate_text_area_ratio(ocr_data, pil_image.size)
            
            # 한글/영어 비율 분석
            korean_ratio, english_ratio = self._analyze_language_ratio(cleaned_text)
            
            return {
                'text': cleaned_text,
                'text_length': len(cleaned_text),
                'avg_confidence': avg_confidence / 100.0,  # 0-1 범위로 정규화
                'text_area_ratio': text_area_ratio,
                'korean_ratio': korean_ratio,
                'english_ratio': english_ratio,
                'has_meaningful_text': len(cleaned_text) >= self.min_text_length,
                'word_count': len(cleaned_text.split()) if cleaned_text else 0
            }
            
        except Exception as e:
            logger.warning(f"OCR 분석 실패: {str(e)}")
            return {
                'text': '', 'text_length': 0, 'avg_confidence': 0.0,
                'text_area_ratio': 0.0, 'korean_ratio': 0.0, 'english_ratio': 0.0,
                'has_meaningful_text': False, 'word_count': 0
            }
    
    def _analyze_color_characteristics(self, image: np.ndarray) -> Dict[str, float]:
        """색상 특성 분석"""
        # PIL 이미지로 변환 (통계 계산용)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 기본 통계
        stat = ImageStat.Stat(pil_image)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 색상 분산 계산 (낮을수록 단순한 색상 = 텍스트 가능성)
        color_variance = np.var(gray)
        
        # 히스토그램 분석
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_variance = np.var(hist)
        
        # 이진화 임계값에서의 픽셀 분포 분석
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_pixel_ratio = np.sum(binary == 255) / binary.size
        black_pixel_ratio = np.sum(binary == 0) / binary.size
        
        # 명도 분석
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        return {
            'color_variance': color_variance,
            'hist_variance': hist_variance,
            'white_pixel_ratio': white_pixel_ratio,
            'black_pixel_ratio': black_pixel_ratio,
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'is_mostly_bw': white_pixel_ratio + black_pixel_ratio > 0.8  # 대부분 흑백
        }
    
    def _analyze_edge_characteristics(self, image: np.ndarray) -> Dict[str, float]:
        """엣지 특성 분석 (텍스트는 많은 수직/수평 엣지를 가짐)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Sobel 엣지 (수직/수평 방향별로)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 수직/수평 엣지 강도
        vertical_edges = np.mean(np.abs(sobel_x))
        horizontal_edges = np.mean(np.abs(sobel_y))
        
        # 엣지 방향성 분석 (텍스트는 수직/수평 엣지가 많음)
        edge_orientation_score = (vertical_edges + horizontal_edges) / (np.mean(np.abs(sobel_x + sobel_y)) + 1e-6)
        
        # 라인 검출 (Hough Transform)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        line_count = len(lines) if lines is not None else 0
        
        return {
            'edge_density': edge_density,
            'vertical_edges': vertical_edges,
            'horizontal_edges': horizontal_edges,
            'edge_orientation_score': min(edge_orientation_score, 2.0),  # 상한 제한
            'line_count': line_count,
            'has_strong_lines': line_count > 10  # 강한 선이 많으면 텍스트 가능성
        }
    
    def _analyze_texture_complexity(self, image: np.ndarray) -> Dict[str, float]:
        """텍스처 복잡도 분석"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 지역적 표준편차 계산 (텍스처 복잡도 측정)
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_variance = local_sqr_mean - local_mean**2
        texture_complexity = np.mean(local_variance)
        
        # 라플라시안 분산 (이미지 선명도/복잡도)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 엔트로피 계산 (정보 복잡도)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # 정규화
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # 패턴 반복성 분석 (템플릿 매칭으로 반복 패턴 검출)
        # 작은 패치들의 유사성을 확인
        h, w = gray.shape
        patch_size = min(20, h//10, w//10)
        if patch_size >= 5:
            template = gray[h//4:h//4+patch_size, w//4:w//4+patch_size]
            match_result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            max_matches = np.sum(match_result > 0.8)  # 높은 유사도 매치 개수
            pattern_repetition = max_matches / (match_result.size + 1)
        else:
            pattern_repetition = 0.0
        
        return {
            'texture_complexity': texture_complexity,
            'laplacian_variance': laplacian_var,
            'entropy': entropy,
            'pattern_repetition': pattern_repetition,
            'is_low_complexity': texture_complexity < 100  # 낮은 복잡도 = 텍스트 가능성
        }
    
    def _calculate_text_area_ratio(self, ocr_data: Dict, image_size: Tuple[int, int]) -> float:
        """OCR 결과에서 텍스트가 차지하는 영역 비율 계산"""
        try:
            total_area = image_size[0] * image_size[1]
            text_area = 0
            
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 30:  # 신뢰도가 30 이상인 텍스트만
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    text_area += w * h
            
            return min(text_area / total_area, 1.0)
            
        except Exception:
            return 0.0
    
    def _analyze_language_ratio(self, text: str) -> Tuple[float, float]:
        """텍스트에서 한글과 영어의 비율 분석"""
        if not text:
            return 0.0, 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text)
        
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0.0
        english_ratio = english_chars / total_chars if total_chars > 0 else 0.0
        
        return korean_ratio, english_ratio
    
    def _clean_ocr_text(self, raw_text: str) -> str:
        """OCR 결과 텍스트 정리"""
        if not raw_text:
            return ""
        
        # 불필요한 공백 및 특수문자 정리
        cleaned = re.sub(r'\s+', ' ', raw_text.strip())
        cleaned = re.sub(r'[^\w\s가-힣.,!?()[\]{}-]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _make_classification_decision(self, ocr_result: Dict, color_analysis: Dict, 
                                   edge_analysis: Dict, texture_analysis: Dict) -> Dict[str, any]:
        """종합 분석 결과를 바탕으로 분류 결정"""
        
        # 각 요소별 점수 계산 (0-1 범위, 높을수록 텍스트 가능성)
        scores = {}
        factors = {}
        
        # 1. OCR 텍스트 점수
        if ocr_result['has_meaningful_text'] and ocr_result['avg_confidence'] > 0.3:
            ocr_score = min(ocr_result['text_area_ratio'] * 2.0, 1.0)
            scores['ocr'] = ocr_score
            factors['ocr'] = f"텍스트 {ocr_result['text_length']}자, 신뢰도 {ocr_result['avg_confidence']:.2f}"
        else:
            scores['ocr'] = 0.0
            factors['ocr'] = "의미있는 텍스트 없음"
        
        # 2. 색상 특성 점수 (흑백에 가까울수록, 분산이 낮을수록 텍스트 가능성)
        color_score = 0.0
        if color_analysis['is_mostly_bw']:
            color_score += 0.4
        if color_analysis['color_variance'] < self.color_variance_threshold:
            color_score += 0.4
        if color_analysis['white_pixel_ratio'] > 0.7:  # 배경이 주로 흰색
            color_score += 0.2
        scores['color'] = color_score
        factors['color'] = f"흑백비율 {color_analysis['white_pixel_ratio']+color_analysis['black_pixel_ratio']:.2f}, 분산 {color_analysis['color_variance']:.0f}"
        
        # 3. 엣지 특성 점수 (수직/수평 엣지가 많을수록 텍스트 가능성)
        edge_score = 0.0
        if edge_analysis['edge_density'] > self.edge_density_threshold:
            edge_score += 0.3
        if edge_analysis['has_strong_lines']:
            edge_score += 0.3
        if edge_analysis['edge_orientation_score'] > 1.2:  # 수직/수평 엣지 우세
            edge_score += 0.4
        scores['edge'] = edge_score
        factors['edge'] = f"엣지밀도 {edge_analysis['edge_density']:.3f}, 선 {edge_analysis['line_count']}개"
        
        # 4. 텍스처 복잡도 점수 (복잡도가 낮을수록 텍스트 가능성)
        texture_score = 0.0
        if texture_analysis['is_low_complexity']:
            texture_score += 0.4
        if texture_analysis['pattern_repetition'] > 0.1:  # 반복 패턴 존재
            texture_score += 0.3
        if texture_analysis['entropy'] < 6.0:  # 낮은 엔트로피
            texture_score += 0.3
        scores['texture'] = texture_score
        factors['texture'] = f"복잡도 {texture_analysis['texture_complexity']:.0f}, 엔트로피 {texture_analysis['entropy']:.1f}"
        
        # 가중치 적용한 최종 점수 계산
        weights = {'ocr': 0.4, 'color': 0.2, 'edge': 0.2, 'texture': 0.2}
        final_score = sum(scores[key] * weights[key] for key in scores)
        
        # 분류 결정
        is_text_only = final_score >= self.text_ratio_threshold
        confidence = final_score if is_text_only else (1.0 - final_score)
        
        # 추가 규칙 기반 검증
        if is_text_only:
            # 텍스트로 분류되었지만 실제로는 이미지일 수 있는 경우들을 확인
            if (ocr_result['text_length'] < self.min_text_length or 
                ocr_result['avg_confidence'] < 0.2):
                is_text_only = False
                confidence = 0.3
                factors['override'] = "텍스트 길이/신뢰도 부족으로 이미지 분류"
        
        return {
            'is_text_only': is_text_only,
            'confidence': confidence,
            'factors': factors,
            'detailed_scores': scores
        }
    
    def _make_basic_classification_decision(self, ocr_result: Dict) -> Dict[str, any]:
        """OCR 결과만으로 기본 분류 결정 (OpenCV 없음)"""
        factors = {}
        
        # OCR 텍스트 기반 판단
        has_meaningful_text = ocr_result['has_meaningful_text']
        text_length = ocr_result['text_length']
        avg_confidence = ocr_result['avg_confidence']
        
        # 기본 점수 계산
        if has_meaningful_text and avg_confidence > 0.3 and text_length >= self.min_text_length:
            # 텍스트 전용으로 분류
            is_text_only = True
            confidence = min(0.8, avg_confidence + (text_length / 200) * 0.2)  # 텍스트 길이에 따라 신뢰도 조정
            factors['ocr'] = f"텍스트 {text_length}자, OCR 신뢰도 {avg_confidence:.2f}"
            factors['classification'] = "OCR 기반 텍스트 분류"
        else:
            # 실제 이미지로 분류
            is_text_only = False
            confidence = 0.7  # 기본 이미지 분류 신뢰도
            factors['ocr'] = f"텍스트 부족 ({text_length}자) 또는 낮은 신뢰도 ({avg_confidence:.2f})"
            factors['classification'] = "이미지로 분류"
        
        # 추가 휴리스틱 검사
        if text_length > 100 and avg_confidence > 0.5:
            # 충분한 텍스트와 높은 신뢰도
            is_text_only = True
            confidence = min(0.9, confidence + 0.1)
            factors['boost'] = "긴 텍스트 + 높은 신뢰도"
        elif text_length < 20 or avg_confidence < 0.2:
            # 매우 적은 텍스트 또는 낮은 신뢰도
            is_text_only = False
            confidence = max(0.6, confidence)
            factors['penalty'] = "짧은 텍스트 또는 낮은 OCR 신뢰도"
        
        return {
            'is_text_only': is_text_only,
            'confidence': confidence,
            'factors': factors
        }
    
    def _default_result(self) -> Dict[str, any]:
        """오류 발생 시 기본 결과"""
        return {
            'is_text_only': False,
            'confidence': 0.0,
            'text_content': '',
            'analysis_details': {
                'error': True,
                'message': '이미지 분석 실패'
            }
        }


# 편의 함수들
def classify_single_image(image_path: str, **classifier_kwargs) -> Dict[str, any]:
    """단일 이미지 분류 편의 함수"""
    classifier = ImageClassifier(**classifier_kwargs)
    return classifier.classify_image(image_path)


def batch_classify_images(image_paths: list, **classifier_kwargs) -> Dict[str, Dict[str, any]]:
    """다중 이미지 일괄 분류 편의 함수"""
    classifier = ImageClassifier(**classifier_kwargs)
    results = {}
    
    for image_path in image_paths:
        try:
            results[str(image_path)] = classifier.classify_image(str(image_path))
        except Exception as e:
            logger.error(f"이미지 {image_path} 분류 실패: {str(e)}")
            results[str(image_path)] = classifier._default_result()
    
    return results


def extract_text_from_text_images(classification_results: Dict[str, Dict[str, any]]) -> Dict[str, str]:
    """텍스트 전용으로 분류된 이미지들에서 텍스트 추출"""
    text_results = {}
    
    for image_path, result in classification_results.items():
        if result['is_text_only'] and result['text_content']:
            text_results[image_path] = result['text_content']
    
    return text_results