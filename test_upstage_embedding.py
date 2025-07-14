#!/usr/bin/env python3
"""
업스테이지 solar-embedding-1-large-query 모델 테스트
"""

import os
import sys
import logging
from typing import List

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.embeddings.embedding_model import EmbeddingModel
from config import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_upstage_embedding():
    """업스테이지 임베딩 모델 테스트"""
    print("=" * 60)
    print("🚀 업스테이지 solar-embedding-1-large-query 모델 테스트")
    print("=" * 60)
    
    # API 키 확인
    if not settings.upstage_api_key:
        print("❌ UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("💡 .env 파일에 UPSTAGE_API_KEY=your_api_key 를 추가하세요.")
        return False
    
    try:
        # 임베딩 모델 초기화
        print("📝 임베딩 모델 초기화 중...")
        embedding_model = EmbeddingModel()
        
        # 모델 정보 출력
        model_info = embedding_model.get_model_info()
        print(f"✅ 모델 정보:")
        print(f"   - 타입: {model_info['type']}")
        print(f"   - 모델명: {model_info['model_name']}")
        print(f"   - 차원: {model_info['dimension']}")
        
        # 테스트 텍스트들
        test_texts = [
            "안녕하세요. 이것은 한국어 텍스트입니다.",
            "Hello, this is an English text.",
            "RAG 시스템을 위한 문서 임베딩 테스트",
            "Document embedding test for RAG system"
        ]
        
        print("\n📊 단일 쿼리 임베딩 테스트:")
        for i, text in enumerate(test_texts[:2], 1):
            print(f"   {i}. 텍스트: '{text}'")
            embedding = embedding_model.embed_query(text)
            print(f"      임베딩 차원: {len(embedding)}")
            print(f"      임베딩 샘플: {embedding[:5]} ... (처음 5개 값)")
        
        print("\n📚 문서 배치 임베딩 테스트:")
        embeddings = embedding_model.embed_documents(test_texts)
        print(f"   총 문서 수: {len(embeddings)}")
        print(f"   각 임베딩 차원: {len(embeddings[0])}")
        
        # 유사도 계산 테스트
        print("\n🔍 유사도 계산 테스트:")
        import numpy as np
        
        # 첫 번째와 세 번째 텍스트 (둘 다 한국어 관련)
        emb1 = np.array(embeddings[0])
        emb3 = np.array(embeddings[2])
        
        # 코사인 유사도 계산
        cosine_sim = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        print(f"   한국어 텍스트 간 유사도: {cosine_sim:.4f}")
        
        # 첫 번째와 두 번째 텍스트 (서로 다른 언어)
        emb2 = np.array(embeddings[1])
        cosine_sim2 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"   한국어-영어 텍스트 간 유사도: {cosine_sim2:.4f}")
        
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {str(e)}")
        logger.error(f"테스트 오류: {str(e)}", exc_info=True)
        return False

def show_setup_guide():
    """설정 가이드 출력"""
    print("\n" + "=" * 60)
    print("📋 업스테이지 임베딩 모델 설정 가이드")
    print("=" * 60)
    print("1. 업스테이지 콘솔에서 API 키를 발급받으세요:")
    print("   https://console.upstage.ai/")
    print()
    print("2. .env 파일에 API 키를 추가하세요:")
    print("   UPSTAGE_API_KEY=your_api_key_here")
    print()
    print("3. 필요한 패키지를 설치하세요:")
    print("   pip install langchain-upstage")
    print()
    print("4. 현재 설정:")
    print(f"   - 임베딩 제공자: {settings.embedding_provider}")
    print(f"   - 업스테이지 모델: {settings.upstage_embedding_model}")
    print(f"   - API 키 설정됨: {'✅' if settings.upstage_api_key else '❌'}")

if __name__ == "__main__":
    show_setup_guide()
    
    if settings.embedding_provider == "upstage":
        print("\n🧪 테스트 시작...")
        success = test_upstage_embedding()
        
        if success:
            print("\n🎉 업스테이지 임베딩 모델이 성공적으로 설정되었습니다!")
        else:
            print("\n😞 테스트가 실패했습니다. 설정을 다시 확인해주세요.")
    else:
        print(f"\n⚠️  현재 임베딩 제공자가 '{settings.embedding_provider}'로 설정되어 있습니다.")
        print("config.py에서 embedding_provider를 'upstage'로 변경하세요.") 