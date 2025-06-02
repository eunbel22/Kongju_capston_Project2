from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import asyncio
from typing import List, Dict, Optional
import uvicorn
import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from bs4 import BeautifulSoup

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="공주대학교 AI 서버", version="1.0.0")

# 절대 경로 설정
PROJECT_ROOT = "/home/porty/Porty/PortyProject"
DATA_FOLDER = os.path.join(PROJECT_ROOT, "back", "data")
LOG_FOLDER = os.path.join(PROJECT_ROOT, "server", "logs")

# 로그 폴더 생성
os.makedirs(LOG_FOLDER, exist_ok=True)


# 요청/응답 모델 정의
class ChatMessage(BaseModel):
    role: str  # "user" 또는 "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "mistral"
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    model: str
    created_at: str


# Ollama 설정
OLLAMA_URL = "http://localhost:11434"


# RAG 시스템 클래스
class RAGSystem:
    def __init__(self, data_folder=None):
        self.data_folder = data_folder or DATA_FOLDER
        self.documents = []
        self.embeddings = None
        self.model = None
        self.load_model()
        self.load_documents()
        self.create_embeddings()

    def load_model(self):
        """임베딩 모델 로드"""
        try:
            # 한국어 지원 임베딩 모델 사용
            logger.info("임베딩 모델 로드 시작...")
            self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            logger.info("임베딩 모델 로드 완료")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            try:
                # 대안 모델 사용
                logger.info("대안 임베딩 모델 로드 시도...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("대안 임베딩 모델 로드 완료")
            except Exception as e2:
                logger.error(f"대안 모델 로드도 실패: {e2}")
                self.model = None

    def load_documents(self):
        """JSON 파일들을 로드하여 문서 데이터베이스 구성"""
        logger.info(f"데이터 폴더 확인: {self.data_folder}")

        if not os.path.exists(self.data_folder):
            logger.warning(f"데이터 폴더가 존재하지 않습니다: {self.data_folder}")
            # 데이터 폴더 생성 시도
            try:
                os.makedirs(self.data_folder, exist_ok=True)
                logger.info(f"데이터 폴더 생성: {self.data_folder}")
            except Exception as e:
                logger.error(f"데이터 폴더 생성 실패: {e}")
            return

        json_files = glob.glob(os.path.join(self.data_folder, "*.json"))
        logger.info(f"발견된 JSON 파일: {len(json_files)}개")

        for file_path in json_files:
            try:
                logger.debug(f"파일 로드 중: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.process_json_data(data, os.path.basename(file_path))
            except Exception as e:
                logger.error(f"파일 로드 실패 {file_path}: {e}")

        logger.info(f"총 {len(self.documents)}개의 문서 로드 완료")

    def process_json_data(self, data, filename):
        """공주대학교 웹사이트 크롤링 JSON 데이터 처리"""
        if isinstance(data, dict) and "menu" in data and "content" in data:
            # 웹사이트 크롤링 데이터 형식 처리
            menu = data["menu"]
            content = data["content"]

            # HTML 태그 제거 및 텍스트 정리
            cleaned_content = self.clean_html_content(content)

            # 의미있는 섹션들로 분할
            sections = self.extract_meaningful_sections(cleaned_content, menu)

            for section in sections:
                if section["content"].strip():  # 빈 내용 제외
                    doc = {
                        "content": section["content"],
                        "source": filename,
                        "category": menu,
                        "section": section["title"],
                        "metadata": {
                            "file": filename,
                            "menu": menu,
                            "section": section["title"],
                            "type": "website_content"
                        }
                    }
                    self.documents.append(doc)

        elif isinstance(data, list):
            # 리스트 형태의 데이터 처리 (기존 방식 유지)
            for i, item in enumerate(data):
                if isinstance(item, dict) and "menu" in item and "content" in item:
                    self.process_json_data(item, f"{filename}[{i}]")

        else:
            # 기존 방식으로 처리 (이전 JSON 형식 호환성)
            self.process_legacy_json_data(data, filename)

    def clean_html_content(self, content):
        """HTML 태그 제거 및 텍스트 정리"""
        try:
            # BeautifulSoup으로 HTML 파싱
            soup = BeautifulSoup(content, 'html.parser')

            # 불필요한 태그 제거
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()

            # 텍스트 추출
            text = soup.get_text()

            # 텍스트 정리
            text = re.sub(r'\n+', '\n', text)  # 연속된 줄바꿈 제거
            text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
            text = re.sub(r'[^\w\s가-힣.,!?()[\]{}:;-]', '', text)  # 특수문자 제거

            return text.strip()
        except Exception as e:
            logger.warning(f"HTML 정리 실패: {e}")
            # BeautifulSoup 실패시 정규식으로 대체
            text = re.sub(r'<[^>]+>', '', content)  # HTML 태그 제거
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    def extract_meaningful_sections(self, content, menu):
        """의미있는 섹션들로 텍스트 분할"""
        sections = []

        # 메뉴별 특별 처리
        if "업무추진비" in menu:
            sections.extend(self.extract_budget_sections(content, menu))
        elif "학사" in menu or "일정" in menu:
            sections.extend(self.extract_academic_sections(content, menu))
        elif "식단" in menu or "메뉴" in menu:
            sections.extend(self.extract_dining_sections(content, menu))
        elif "캠퍼스" in menu or "찾아오시는" in menu:
            sections.extend(self.extract_campus_sections(content, menu))
        else:
            # 일반적인 섹션 분할
            sections.extend(self.extract_general_sections(content, menu))

        return sections

    def extract_budget_sections(self, content, menu):
        """업무추진비 관련 섹션 추출"""
        sections = []

        # 월별 업무추진비 내역 추출
        budget_pattern = r'(\d{4}년\s*\d{1,2}월.*?업무추진비.*?집행.*?내역)'
        matches = re.findall(budget_pattern, content, re.IGNORECASE)

        for match in matches:
            sections.append({
                "title": "월별 업무추진비 내역",
                "content": match
            })

        # 전체 내용도 하나의 섹션으로 추가
        if not sections:
            sections.append({
                "title": "업무추진비 정보",
                "content": content[:1000]  # 처음 1000자만
            })

        return sections

    def extract_academic_sections(self, content, menu):
        """학사 관련 섹션 추출"""
        sections = []

        # 학사일정 관련 키워드로 섹션 분할
        keywords = ["개강", "종강", "중간고사", "기말고사", "수강신청", "성적", "휴학", "복학"]

        for keyword in keywords:
            pattern = rf'([^.]*{keyword}[^.]*\.)'
            matches = re.findall(pattern, content, re.IGNORECASE)

            if matches:
                section_content = ' '.join(matches)
                sections.append({
                    "title": f"{keyword} 관련 정보",
                    "content": section_content
                })

        # 전체 내용도 추가
        sections.append({
            "title": "학사 종합 정보",
            "content": content[:1500]
        })

        return sections

    def extract_dining_sections(self, content, menu):
        """식단 관련 섹션 추출"""
        sections = []

        # 캠퍼스별, 시간대별 식단 정보 추출
        campus_keywords = ["공주", "천안", "예산"]
        meal_keywords = ["아침", "점심", "저녁", "조식", "중식", "석식"]

        for campus in campus_keywords:
            campus_content = []
            for meal in meal_keywords:
                pattern = rf'([^.]*{campus}[^.]*{meal}[^.]*\.)'
                matches = re.findall(pattern, content, re.IGNORECASE)
                campus_content.extend(matches)

            if campus_content:
                sections.append({
                    "title": f"{campus}캠퍼스 식단",
                    "content": ' '.join(campus_content)
                })

        return sections

    def extract_campus_sections(self, content, menu):
        """캠퍼스 정보 섹션 추출"""
        sections = []

        # 캠퍼스별 정보 추출
        campus_info = {
            "공주캠퍼스": ["공주시", "공주대학로", "041-850"],
            "천안캠퍼스": ["천안시", "천안대로", "041-521"],
            "예산캠퍼스": ["예산군", "예산읍", "041-330"]
        }

        for campus, keywords in campus_info.items():
            campus_content = []
            for keyword in keywords:
                pattern = rf'([^.]*{keyword}[^.]*\.)'
                matches = re.findall(pattern, content, re.IGNORECASE)
                campus_content.extend(matches)

            if campus_content:
                sections.append({
                    "title": campus,
                    "content": ' '.join(campus_content)
                })

        return sections

    def extract_general_sections(self, content, menu):
        """일반적인 섹션 분할"""
        sections = []

        # 문장 단위로 분할 (너무 긴 내용 방지)
        sentences = re.split(r'[.!?]\s+', content)

        # 의미있는 길이의 문장들을 그룹화
        current_section = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > 500:  # 500자 단위로 섹션 분할
                if current_section:
                    sections.append({
                        "title": f"{menu} 정보",
                        "content": '. '.join(current_section) + '.'
                    })
                current_section = [sentence]
                current_length = len(sentence)
            else:
                current_section.append(sentence)
                current_length += len(sentence)

        # 마지막 섹션 추가
        if current_section:
            sections.append({
                "title": f"{menu} 정보",
                "content": '. '.join(current_section) + '.'
            })

        return sections

    def process_legacy_json_data(self, data, filename):
        """기존 JSON 형식 처리 (호환성 유지)"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    doc = {
                        "content": f"{key}: {value}",
                        "source": filename,
                        "category": key,
                        "metadata": {"file": filename, "key": key, "type": "legacy"}
                    }
                    self.documents.append(doc)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        doc = {
                            "content": f"{key} - {sub_key}: {sub_value}",
                            "source": filename,
                            "category": key,
                            "metadata": {"file": filename, "key": f"{key}.{sub_key}", "type": "legacy"}
                        }
                        self.documents.append(doc)

    def create_embeddings(self):
        """문서들의 임베딩 생성"""
        if not self.documents or not self.model:
            logger.warning("문서나 모델이 없어 임베딩을 생성할 수 없습니다")
            return

        try:
            contents = [doc["content"] for doc in self.documents]
            logger.info(f"임베딩 생성 시작: {len(contents)}개 문서")
            self.embeddings = self.model.encode(contents)
            logger.info(f"임베딩 생성 완료: {self.embeddings.shape}")
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")

    def search_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """쿼리와 유사한 문서 검색"""
        if self.embeddings is None or not self.model:
            logger.warning("임베딩이나 모델이 없어 검색할 수 없습니다")
            return []

        try:
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query])

            # 코사인 유사도 계산
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]

            # 상위 k개 문서 인덱스 찾기
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # 결과 반환
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # 최소 유사도 임계값
                    doc = self.documents[idx].copy()
                    doc["similarity"] = float(similarities[idx])
                    results.append(doc)

            return results
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []


# RAG 시스템 초기화
logger.info("RAG 시스템 초기화 시작...")
rag_system = RAGSystem()
logger.info("RAG 시스템 초기화 완료")


@app.get("/")
async def root():
    return {
        "message": "공주대학교 AI 서버가 실행 중입니다",
        "documents_loaded": len(rag_system.documents),
        "data_folder": DATA_FOLDER,
        "log_folder": LOG_FOLDER
    }


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    try:
        # Ollama 서버 상태 확인
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "disconnected"

        return {
            "status": "healthy",
            "ollama": ollama_status,
            "rag_documents": len(rag_system.documents),
            "rag_embeddings": rag_system.embeddings is not None,
            "data_folder": DATA_FOLDER,
            "data_folder_exists": os.path.exists(DATA_FOLDER)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/api/rag/search")
async def search_documents(query: str, top_k: int = 3):
    """RAG 문서 검색 테스트 엔드포인트"""
    try:
        results = rag_system.search_similar_documents(query, top_k)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """AI와 채팅하기 (RAG 적용)"""
    try:
        # 사용자의 마지막 메시지 추출
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        # RAG 검색 수행
        relevant_docs = rag_system.search_similar_documents(user_message, top_k=3)

        # 검색된 문서들을 컨텍스트로 구성
        context = ""
        if relevant_docs:
            context = "\n\n=== 관련 정보 ===\n"
            for i, doc in enumerate(relevant_docs, 1):
                context += f"{i}. {doc['content']} (출처: {doc['source']})\n"
            context += "=== 관련 정보 끝 ===\n\n"

        # 공주대학교 특화 시스템 프롬프트 (RAG 컨텍스트 포함)
        system_prompt = f"""당신은 공주대학교 학생들을 위한 AI 어시스턴트 '포티'입니다. 
다음 역할을 수행해주세요:
1. 공주대학교 관련 정보 제공 (학사일정, 캠퍼스 정보, 식단 등)
2. 학습 도움 (과제, 시험 준비, 학습 방법 등)
3. 대학생활 상담 (동아리, 진로, 취업 등)
4. 항상 친근하고 도움이 되는 톤으로 답변
5. 한국어로 답변

공주대학교는 공주캠퍼스, 천안캠퍼스, 예산캠퍼스가 있습니다.

{context}

위의 관련 정보가 있다면 이를 참고하여 정확하고 구체적인 답변을 제공해주세요. 
관련 정보가 없거나 부족하다면 일반적인 도움말을 제공해주세요."""

        # 메시지 구성
        messages = [{"role": "system", "content": system_prompt}]

        # 사용자 메시지 추가
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        # Ollama API 호출
        ollama_request = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_gpu": -1,
                "num_thread": -1
            }
        }

        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=ollama_request,
            timeout=60
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="AI 서버 오류가 발생했습니다")

        result = response.json()

        # 응답에 RAG 정보 추가 (디버깅용)
        response_data = ChatResponse(
            response=result["message"]["content"],
            model=request.model,
            created_at=result.get("created_at", "")
        )

        # RAG 정보를 로그에 기록
        if relevant_docs:
            logger.info(f"RAG 검색 결과: {len(relevant_docs)}개 문서 사용")
            for doc in relevant_docs:
                logger.info(f"- {doc['source']}: {doc['similarity']:.3f}")

        return response_data

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="AI 응답 시간이 초과되었습니다")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="AI 서버에 연결할 수 없습니다")
    except Exception as e:
        logger.error(f"채팅 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


@app.get("/api/models")
async def get_available_models():
    """사용 가능한 모델 목록 조회"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=500, detail="모델 목록을 가져올 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/reload")
async def reload_rag_data():
    """RAG 데이터 다시 로드"""
    try:
        global rag_system
        logger.info("RAG 시스템 다시 로드 시작...")
        rag_system = RAGSystem()
        logger.info("RAG 시스템 다시 로드 완료")
        return {
            "message": "RAG 데이터 다시 로드 완료",
            "documents_loaded": len(rag_system.documents),
            "data_folder": DATA_FOLDER
        }
    except Exception as e:
        logger.error(f"RAG 데이터 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"RAG 데이터 로드 실패: {str(e)}")


if __name__ == "__main__":
    # GPU 최적화를 위한 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 첫 번째 GPU 사용

    logger.info(f"AI 서버 시작 준비...")
    logger.info(f"프로젝트 루트: {PROJECT_ROOT}")
    logger.info(f"데이터 폴더: {DATA_FOLDER}")
    logger.info(f"로그 폴더: {LOG_FOLDER}")

    uvicorn.run(
        app,
        host="127.0.0.1",  # 내부 통신만 허용
        port=8000,
        workers=1,  # GPU 메모리 절약을 위해 단일 워커
        log_level="info"
    )
