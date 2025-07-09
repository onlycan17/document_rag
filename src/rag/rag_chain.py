from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from config import settings
from src.vectorstore import VectorDatabase
import requests
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, vector_db: Optional[VectorDatabase] = None):
        # 벡터 DB 인스턴스를 외부에서 주입받거나 새로 생성
        if vector_db is not None:
            self.vector_db = vector_db
        else:
            self.vector_db = VectorDatabase()
        
        self.llm = self._initialize_llm(provider, model)
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()
        self.current_provider = provider or settings.llm_provider
        self.current_model = model
    
    def _initialize_llm(self, provider: Optional[str] = None, model: Optional[str] = None):
        """LLM 초기화"""
        # provider가 지정되지 않으면 설정에서 가져옴
        if provider is None:
            provider = settings.llm_provider
            
        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            return ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model_name=model or settings.openai_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
        
        elif provider == "google":
            if not settings.google_api_key:
                raise ValueError("Google API 키가 설정되지 않았습니다.")
            return ChatGoogleGenerativeAI(
                google_api_key=settings.google_api_key,
                model=model or settings.google_model,
                temperature=settings.temperature,
                max_output_tokens=settings.max_tokens
            )
        
        elif provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
            return ChatAnthropic(
                anthropic_api_key=settings.anthropic_api_key,
                model_name=model or settings.anthropic_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
        
        elif provider == "local":
            # OpenAI 호환 API를 사용하는 로컬 모델
            return ChatOpenAI(
                openai_api_key=settings.local_llm_api_key,
                openai_api_base=settings.local_llm_base_url + "/v1",
                model_name=model or settings.local_llm_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
        
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""
        template = """당신은 복잡한 내용을 쉽게 설명하는 친절한 AI 어시스턴트입니다. 
전문 용어나 어려운 개념을 일반인도 이해할 수 있도록 쉽게 풀어서 설명하는 것이 당신의 역할입니다.

다음은 검색된 관련 문서들입니다:
{context}

위 문서들의 내용을 참고하여 다음 질문에 답변해주세요:
{question}

답변 시 반드시 지켜야 할 규칙:

1. **쉬운 설명 제공**:
   - 전문 용어가 나오면 반드시 쉬운 말로 풀어서 설명하세요
   - 복잡한 개념은 일상생활의 예시를 들어 설명하세요
   - 어려운 단어는 괄호 안에 쉬운 설명을 추가하세요
   예: "전자정부법(정부가 온라인으로 서비스를 제공하는 것에 관한 법)"

2. **구조화된 답변**:
   - 중요한 내용은 번호나 불릿 포인트로 정리하세요
   - 긴 설명은 단락을 나누어 읽기 쉽게 만드세요
   - 핵심 내용을 먼저 요약하고, 그 다음 상세 설명을 제공하세요

3. **예시 활용**:
   - 추상적인 개념은 구체적인 예시를 들어 설명하세요
   - "예를 들어", "쉽게 말하면", "다시 말해" 등의 표현을 활용하세요
   - 일반인이 경험할 수 있는 상황에 비유하여 설명하세요

4. **정확성 유지**:
   - 제공된 문서의 내용만을 기반으로 답변하세요
   - 문서에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요
   - 법률이나 규정 관련 내용은 정확히 인용하되, 그 의미는 쉽게 풀어서 설명하세요

5. **친근한 어조**:
   - 딱딱한 표현보다는 친근하고 이해하기 쉬운 어조를 사용하세요
   - "~입니다"보다는 "~예요", "~이에요" 같은 부드러운 표현을 사용하세요
   - 독자가 이해했는지 확인하는 듯한 표현을 사용하세요

답변:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _create_chain(self):
        """LLM 체인 생성"""
        # 새로운 LCEL 방식 사용
        return self.prompt_template | self.llm
    
    def _format_documents(self, documents: List[tuple]) -> str:
        """검색된 문서를 컨텍스트로 포맷"""
        context_parts = []
        
        for i, (doc, score) in enumerate(documents):
            # FAISS는 거리 기반이므로 점수가 낮을수록 좋음
            # 모든 검색된 문서를 포함하여 컨텍스트 생성
            
            source = doc.metadata.get('source', '알 수 없음')
            file_name = doc.metadata.get('file_name', '알 수 없음')
            content = doc.page_content.strip()
            
            # 점수 정보도 포함 (디버깅용)
            score_info = f" (점수: {float(score):.4f})" if hasattr(score, '__float__') else ""
            
            context_parts.append(f"[문서 {i+1} - {file_name}{score_info}]\n{content}\n")
        
        return "\n---\n".join(context_parts)
    
    def query(self, question: str) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        try:
            # 벡터 DB 상태 확인
            doc_count = self.vector_db.get_document_count()
            logger.info(f"벡터 DB 문서 수: {doc_count}")
            
            # 1. 관련 문서 검색
            relevant_docs = self.vector_db.search(question, k=settings.k_documents)
            logger.info(f"검색 결과: {len(relevant_docs)}개 문서")
            
            if not relevant_docs:
                if doc_count == 0:
                    return {
                        "answer": "벡터 데이터베이스에 문서가 없습니다. 먼저 문서를 업로드해주세요.",
                        "sources": [],
                        "status": "no_documents"
                    }
                else:
                    # 검색 임계값 때문에 결과가 필터링되었을 수 있음
                    # 더 관대한 검색을 시도
                    try:
                        # FAISS에서 직접 검색 (임계값 무시)
                        if hasattr(self.vector_db.vector_store, 'similarity_search_with_score'):
                            raw_results = self.vector_db.vector_store.similarity_search_with_score(question, k=settings.k_documents)
                            logger.info(f"원시 검색 결과: {len(raw_results)}개, 점수: {[score for _, score in raw_results]}")
                        
                        return {
                            "answer": f"🔍 **검색 결과 분석**\n\n"
                                    f"현재 벡터 데이터베이스에 **{doc_count}개의 문서 청크**가 저장되어 있습니다.\n\n"
                                    f"하지만 질문 '{question}'과 관련된 문서를 찾을 수 없었습니다.\n\n"
                                    f"**가능한 원인:**\n"
                                    f"• 검색 임계값이 너무 엄격할 수 있습니다\n"
                                    f"• 질문과 문서 내용의 용어가 다를 수 있습니다\n"
                                    f"• 문서가 다른 주제일 수 있습니다\n\n"
                                    f"**해결 방법:**\n"
                                    f"• 다른 키워드로 질문해보세요\n"
                                    f"• 더 구체적이거나 더 일반적인 질문을 시도해보세요\n"
                                    f"• 디버그 모드를 켜서 상세 정보를 확인해보세요",
                            "sources": [],
                            "status": "no_relevant_documents"
                        }
                    except Exception as e:
                        logger.error(f"원시 검색 실패: {str(e)}")
                        return {
                            "answer": f"현재 {doc_count}개의 문서 청크가 저장되어 있지만, 질문과 관련된 문서를 찾을 수 없습니다.\n다른 질문을 시도해보세요.",
                            "sources": [],
                            "status": "no_relevant_documents"
                        }
            
            # 2. 컨텍스트 생성
            context = self._format_documents(relevant_docs)
            
            # 3. LLM을 통한 답변 생성
            response = self.chain.invoke({
                "context": context,
                "question": question
            })
            
            # 응답 텍스트 추출 (LCEL 방식)
            if hasattr(response, 'content'):
                answer_text = response.content
            elif isinstance(response, dict) and 'text' in response:
                answer_text = response['text']
            elif isinstance(response, str):
                answer_text = response
            else:
                answer_text = str(response)
            
            # 4. 출처 정보 수집
            sources = []
            for doc, score in relevant_docs:
                source_info = {
                    "file_name": doc.metadata.get('file_name', '알 수 없음'),
                    "source_path": doc.metadata.get('source', '알 수 없음'),
                    "relevance_score": float(score) if hasattr(score, '__float__') else score,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            return {
                "answer": answer_text,
                "sources": sources,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "status": "error"
            }
    
    def update_llm(self, provider: str, model: Optional[str] = None):
        """LLM 제공자 및 모델 변경"""
        self.llm = self._initialize_llm(provider, model)
        self.chain = self._create_chain()
        self.current_provider = provider
        self.current_model = model
        
    def get_available_models(self) -> Dict[str, List[Dict[str, str]]]:
        """사용 가능한 모델 목록 반환"""
        models = {
            "openai": [
                {"name": "GPT-3.5 Turbo", "model": "gpt-3.5-turbo", "description": "빠르고 효율적"},
                {"name": "GPT-4", "model": "gpt-4", "description": "더 정확하지만 느림"},
                {"name": "GPT-4 Turbo", "model": "gpt-4-turbo-preview", "description": "GPT-4의 빠른 버전"},
                {"name": "GPT-4o", "model": "gpt-4o", "description": "최신 옴니 모델"},
                {"name": "GPT-4o mini", "model": "gpt-4o-mini", "description": "가벼운 옴니 모델"}
            ],
            "google": [
                {"name": "Gemini 1.5 Flash", "model": "gemini-1.5-flash", "description": "빠른 응답 (8K 컨텍스트)"},
                {"name": "Gemini 1.5 Flash-8B", "model": "gemini-1.5-flash-8b", "description": "더 빠른 경량 모델"},
                {"name": "Gemini 1.5 Pro", "model": "gemini-1.5-pro", "description": "고급 기능 (2M 컨텍스트)"},
                {"name": "Gemini 1.0 Pro", "model": "gemini-1.0-pro", "description": "안정적인 버전"}
            ],
            "anthropic": [
                {"name": "Claude 3.5 Sonnet", "model": "claude-3-5-sonnet-20241022", "description": "최신 최고 성능"},
                {"name": "Claude 3 Haiku", "model": "claude-3-haiku-20240307", "description": "빠르고 효율적"},
                {"name": "Claude 3 Sonnet", "model": "claude-3-sonnet-20240229", "description": "균형잡힌 성능"},
                {"name": "Claude 3 Opus", "model": "claude-3-opus-20240229", "description": "최고 성능"}
            ],
            "local": [
                {"name": "로컬 모델", "model": "local-model", "description": "현재 실행 중인 모델"}
            ]
        }
        
        # 로컬 모델이 실행 중인 경우, 실제 모델 목록 가져오기
        local_models = self._get_local_models()
        if local_models:
            models["local"] = local_models
        
        return models
    
    def _get_local_models(self) -> List[Dict[str, str]]:
        """로컬 서버에서 사용 가능한 모델 목록 조회"""
        try:
            import requests
            response = requests.get(f"{settings.local_llm_base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                data = response.json()
                local_models = []
                for model in data.get("data", []):
                    local_models.append({
                        "name": model.get("id", "unknown"),
                        "model": model.get("id", "unknown"),
                        "description": f"로컬 모델 - {model.get('owned_by', 'unknown')}"
                    })
                return local_models if local_models else [{"name": "로컬 모델", "model": "local-model", "description": "현재 실행 중인 모델"}]
        except:
            pass
        return []
    
    def add_feedback(self, question: str, answer: str, feedback: str):
        """사용자 피드백 저장 (향후 개선을 위한 기능)"""
        # TODO: 피드백을 데이터베이스에 저장하여 모델 개선에 활용
        pass