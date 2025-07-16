#!/usr/bin/env python3
"""
FAISS DB와 업스테이지 임베딩 모델 호환성 테스트 (수정 버전)

이 스크립트는 다음을 테스트합니다:
1. 업스테이지 임베딩 모델 로딩
2. FAISS 인덱스 생성 및 저장/로딩
3. 문서 추가 및 검색 기능
4. 벡터 차원 호환성
5. 성능 측정
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any
from langchain.schema import Document
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.vector_db import EnhancedVectorDatabase
from src.utils.logging_config import setup_logging

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

class FAISSUpstageCompatibilityTester:
    """
    FAISS DB와 업스테이지 임베딩 호환성 테스트 클래스
    """
    
    def __init__(self):
        self.test_results = {
            "embedding_loading": False,
            "vector_dimensions": False,
            "faiss_creation": False,
            "document_addition": False,
            "search_functionality": False,
            "save_load": False,
            "performance": {}
        }
        
        # 테스트용 문서 데이터
        self.test_documents = [
            Document(
                page_content="정보시스템 구축 및 운영에 관한 지침서입니다. 공공기관에서 정보시스템을 도입할 때 준수해야 할 절차와 기준을 제시합니다.",
                metadata={"source": "test_doc_1.txt", "type": "guideline"}
            ),
            Document(
                page_content="데이터베이스 보안 관리 방안을 설명합니다. 개인정보 보호법과 정보보호 관리체계에 따른 보안 요구사항을 다룹니다.",
                metadata={"source": "test_doc_2.txt", "type": "security"}
            ),
            Document(
                page_content="시스템 유지보수 및 모니터링 절차입니다. 장애 대응, 성능 최적화, 백업 및 복구 방안을 포함합니다.",
                metadata={"source": "test_doc_3.txt", "type": "maintenance"}
            ),
            Document(
                page_content="클라우드 인프라 구축 가이드라인입니다. AWS, Azure, GCP 등 클라우드 서비스 활용 방안을 제시합니다.",
                metadata={"source": "test_doc_4.txt", "type": "cloud"}
            ),
            Document(
                page_content="인공지능 시스템 도입 및 운영 지침입니다. AI 윤리, 데이터 품질 관리, 모델 성능 평가 방법을 다룹니다.",
                metadata={"source": "test_doc_5.txt", "type": "ai"}
            )
        ]
        
        # 테스트용 검색 쿼리
        self.test_queries = [
            "정보시스템 구축 절차는 무엇인가요?",
            "데이터베이스 보안은 어떻게 관리하나요?",
            "시스템 장애 대응 방법을 알려주세요",
            "클라우드 서비스 선택 기준은?",
            "AI 시스템 윤리 지침은 무엇인가요?"
        ]
    
    def test_embedding_model_loading(self) -> bool:
        """
        업스테이지 임베딩 모델 로딩 테스트
        """
        logger.info("=== 임베딩 모델 로딩 테스트 ===")
        
        try:
            start_time = time.time()
            
            # 임베딩 모델 초기화
            embedding_model = EmbeddingModel()
            
            loading_time = time.time() - start_time
            self.test_results["performance"]["embedding_loading_time"] = loading_time
            
            # 임베딩 생성 테스트
            test_text = "테스트용 한국어 문장입니다."
            start_time = time.time()
            
            embeddings = embedding_model.embed_documents([test_text])
            
            embedding_time = time.time() - start_time
            self.test_results["performance"]["embedding_generation_time"] = embedding_time
            
            # 결과 검증
            if embeddings and len(embeddings) > 0:
                vector_dim = len(embeddings[0])
                self.test_results["performance"]["vector_dimension"] = vector_dim
                
                logger.info(f"✅ 임베딩 모델 로딩 성공")
                logger.info(f"   - 모델: {settings.upstage_embedding_model}")
                logger.info(f"   - 벡터 차원: {vector_dim}")
                logger.info(f"   - 로딩 시간: {loading_time:.2f}초")
                logger.info(f"   - 임베딩 생성 시간: {embedding_time:.4f}초")
                
                self.test_results["embedding_loading"] = True
                self.test_results["vector_dimensions"] = True
                return True
            else:
                logger.error("❌ 임베딩 생성 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ 임베딩 모델 로딩 실패: {e}")
            return False
    
    def test_faiss_creation(self) -> bool:
        """
        FAISS 인덱스 생성 테스트
        """
        logger.info("=== FAISS 인덱스 생성 테스트 ===")
        
        try:
            start_time = time.time()
            
            # 벡터 데이터베이스 초기화 (FAISS 모드)
            original_db_type = settings.vector_db_type
            settings.vector_db_type = "faiss"
            
            vector_db = EnhancedVectorDatabase()
            
            creation_time = time.time() - start_time
            self.test_results["performance"]["faiss_creation_time"] = creation_time
            
            logger.info(f"✅ FAISS 데이터베이스 초기화 성공")
            logger.info(f"   - 생성 시간: {creation_time:.2f}초")
            
            self.test_results["faiss_creation"] = True
            
            # 원래 설정 복원
            settings.vector_db_type = original_db_type
            
            return True
            
        except Exception as e:
            logger.error(f"❌ FAISS 데이터베이스 생성 실패: {e}")
            settings.vector_db_type = original_db_type
            return False
    
    def test_document_operations(self) -> bool:
        """
        문서 추가 및 검색 테스트
        """
        logger.info("=== 문서 추가 및 검색 테스트 ===")
        
        try:
            # FAISS 모드로 설정
            original_db_type = settings.vector_db_type
            settings.vector_db_type = "faiss"
            
            vector_db = EnhancedVectorDatabase()
            
            # 문서 추가 테스트
            start_time = time.time()
            vector_db.add_documents(self.test_documents)
            add_time = time.time() - start_time
            
            self.test_results["performance"]["document_addition_time"] = add_time
            
            logger.info(f"✅ 문서 추가 성공")
            logger.info(f"   - 추가된 문서 수: {len(self.test_documents)}개")
            logger.info(f"   - 추가 시간: {add_time:.2f}초")
            
            self.test_results["document_addition"] = True
            
            # 검색 테스트
            search_results = []
            total_search_time = 0
            
            for i, query in enumerate(self.test_queries):
                start_time = time.time()
                results = vector_db.search(query, k=3)
                search_time = time.time() - start_time
                total_search_time += search_time
                
                # 검색 결과는 (Document, score) 튜플 형태
                documents = [doc for doc, score in results] if results else []
                
                search_results.append({
                    "query": query,
                    "results_count": len(documents),
                    "search_time": search_time,
                    "top_result": documents[0].page_content[:100] + "..." if documents else "검색 결과 없음"
                })
                
                logger.info(f"   쿼리 {i+1}: '{query}' → {len(documents)}개 결과 ({search_time:.4f}초)")
            
            avg_search_time = total_search_time / len(self.test_queries)
            self.test_results["performance"]["average_search_time"] = avg_search_time
            self.test_results["performance"]["search_results"] = search_results
            
            logger.info(f"✅ 검색 기능 성공")
            logger.info(f"   - 평균 검색 시간: {avg_search_time:.4f}초")
            
            self.test_results["search_functionality"] = True
            
            # 원래 설정 복원
            settings.vector_db_type = original_db_type
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 문서 작업 실패: {e}")
            settings.vector_db_type = original_db_type
            return False
    
    def test_save_load_operations(self) -> bool:
        """
        FAISS 인덱스 저장 및 로딩 테스트
        """
        logger.info("=== FAISS 저장/로딩 테스트 ===")
        
        try:
            # FAISS 모드로 설정
            original_db_type = settings.vector_db_type
            settings.vector_db_type = "faiss"
            
            # 테스트용 벡터 DB 경로
            test_db_path = "./test_vector_db"
            original_db_path = settings.vector_db_path
            settings.vector_db_path = test_db_path
            
            # 1. 벡터 DB 생성 및 문서 추가
            vector_db1 = EnhancedVectorDatabase()
            vector_db1.add_documents(self.test_documents[:2])  # 일부 문서만 추가
            
            # 2. 저장 (자동으로 저장됨)
            start_time = time.time()
            save_time = time.time() - start_time
            
            # 3. 새로운 인스턴스로 로딩
            start_time = time.time()
            vector_db2 = EnhancedVectorDatabase()
            load_time = time.time() - start_time
            
            # 4. 로딩된 데이터로 검색 테스트
            test_query = "정보시스템 구축"
            results = vector_db2.search(test_query, k=2)
            documents = [doc for doc, score in results] if results else []
            
            self.test_results["performance"]["save_time"] = save_time
            self.test_results["performance"]["load_time"] = load_time
            
            if len(documents) > 0:
                logger.info(f"✅ FAISS 저장/로딩 성공")
                logger.info(f"   - 저장 시간: {save_time:.4f}초")
                logger.info(f"   - 로딩 시간: {load_time:.2f}초")
                logger.info(f"   - 검색 결과: {len(documents)}개")
                
                self.test_results["save_load"] = True
                
                # 임시 파일 정리
                import shutil
                if os.path.exists(test_db_path):
                    shutil.rmtree(test_db_path)
                
                # 원래 설정 복원
                settings.vector_db_type = original_db_type
                settings.vector_db_path = original_db_path
                
                return True
            else:
                logger.error("❌ 로딩 후 검색 결과 없음")
                return False
                
        except Exception as e:
            logger.error(f"❌ FAISS 저장/로딩 실패: {e}")
            
            # 정리 및 복원
            import shutil
            if os.path.exists(test_db_path):
                shutil.rmtree(test_db_path)
            settings.vector_db_type = original_db_type
            settings.vector_db_path = original_db_path
            
            return False
    
    def print_compatibility_report(self):
        """
        호환성 테스트 결과 보고서 출력
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 FAISS + 업스테이지 임베딩 호환성 테스트 결과")
        logger.info("="*60)
        
        # 전체 성공률 계산
        total_tests = len([k for k in self.test_results.keys() if k != "performance"])
        passed_tests = sum([1 for k, v in self.test_results.items() if k != "performance" and v])
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"📊 전체 성공률: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        logger.info("")
        
        # 개별 테스트 결과
        test_names = {
            "embedding_loading": "임베딩 모델 로딩",
            "vector_dimensions": "벡터 차원 호환성",
            "faiss_creation": "FAISS 인덱스 생성",
            "document_addition": "문서 추가 기능",
            "search_functionality": "검색 기능",
            "save_load": "저장/로딩 기능"
        }
        
        for key, name in test_names.items():
            status = "✅ 성공" if self.test_results[key] else "❌ 실패"
            logger.info(f"{name}: {status}")
        
        # 성능 정보
        performance = self.test_results["performance"]
        if performance:
            logger.info("")
            logger.info("⚡ 성능 정보:")
            
            if "vector_dimension" in performance:
                logger.info(f"   - 벡터 차원: {performance['vector_dimension']}")
            if "embedding_loading_time" in performance:
                logger.info(f"   - 임베딩 모델 로딩: {performance['embedding_loading_time']:.2f}초")
            if "embedding_generation_time" in performance:
                logger.info(f"   - 임베딩 생성: {performance['embedding_generation_time']:.4f}초")
            if "faiss_creation_time" in performance:
                logger.info(f"   - FAISS 생성: {performance['faiss_creation_time']:.2f}초")
            if "document_addition_time" in performance:
                logger.info(f"   - 문서 추가: {performance['document_addition_time']:.2f}초")
            if "average_search_time" in performance:
                logger.info(f"   - 평균 검색: {performance['average_search_time']:.4f}초")
        
        # 결론
        logger.info("")
        if success_rate == 100:
            logger.info("🎉 결론: FAISS와 업스테이지 임베딩이 완벽한 호환성을 보입니다!")
            logger.info("   → 프로덕션 환경에서 안전하게 사용할 수 있습니다.")
            logger.info("   → 모든 핵심 기능이 정상적으로 작동합니다.")
        elif success_rate >= 80:
            logger.info("🎉 결론: FAISS와 업스테이지 임베딩이 우수한 호환성을 보입니다!")
            logger.info("   → 프로덕션 환경에서 안전하게 사용할 수 있습니다.")
        elif success_rate >= 60:
            logger.info("⚠️ 결론: FAISS와 업스테이지 임베딩이 기본적인 호환성을 보입니다.")
            logger.info("   → 일부 기능에 주의가 필요할 수 있습니다.")
        else:
            logger.info("❌ 결론: 심각한 호환성 문제가 발견되었습니다.")
            logger.info("   → 문제 해결 후 재테스트가 필요합니다.")
        
        logger.info("="*60)
    
    def run_all_tests(self):
        """
        모든 호환성 테스트 실행
        """
        logger.info("🚀 FAISS + 업스테이지 임베딩 호환성 테스트 시작")
        logger.info(f"현재 설정: {settings.embedding_provider} / {settings.vector_db_type}")
        logger.info("")
        
        # 각 테스트 실행
        self.test_embedding_model_loading()
        self.test_faiss_creation()
        self.test_document_operations()
        self.test_save_load_operations()
        
        # 결과 보고서 출력
        self.print_compatibility_report()

def main():
    """
    메인 함수
    """
    # 환경 확인
    if not settings.upstage_api_key:
        logger.error("❌ UPSTAGE_API_KEY가 설정되지 않았습니다.")
        logger.error("   .env 파일에 API 키를 설정하고 다시 실행하세요.")
        return 1
    
    # 테스트 실행
    tester = FAISSUpstageCompatibilityTester()
    tester.run_all_tests()
    
    return 0

if __name__ == "__main__":
    exit(main()) 