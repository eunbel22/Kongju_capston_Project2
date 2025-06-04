from fastapi import FastAPI, HTTPException  # FastAPI 웹 프레임워크와 예외 처리용 클래스
from pydantic import BaseModel  # 요청/응답 데이터 모델 정의용
import requests  # 외부 API 호출용
import json  # JSON 처리용
import asyncio  # 비동기 처리를 위한 모듈
from typing import List, Dict, Optional  # 타입 힌트용
import uvicorn  # 개발 서버 실행용
import os  # 파일 시스템 경로 처리용
import glob  # 파일 패턴 매칭용
from sentence_transformers import SentenceTransformer  # 임베딩 모델 로드용
import numpy as np  # 수치 계산용
from sklearn.metrics.pairwise import cosine_similarity  # 코사인 유사도 계산용
import logging  # 로깅 설정용
import re  # 정규 표현식 처리용
from bs4 import BeautifulSoup  # HTML 파싱용

# =============================
# 로깅 설정
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# FastAPI 애플리케이션 생성
# =============================
app = FastAPI(title="공주대학교 AI 서버", version="1.0.0")

# =============================
# 절대 경로 설정
# =============================
PROJECT_ROOT = "/home/porty/Porty/PortyProject"  # 프로젝트 루트 디렉터리
DATA_FOLDER = os.path.join(PROJECT_ROOT, "back", "data")  # RAG에 사용할 데이터 폴더
LOG_FOLDER = os.path.join(PROJECT_ROOT, "server", "logs")  # 로그를 저장할 폴더

# 로그 폴더가 없으면 생성
os.makedirs(LOG_FOLDER, exist_ok=True)


# =============================
# 요청/응답 모델 정의
# =============================

class ChatMessage(BaseModel):
    """
    클라이언트가 전송하는 채팅 메시지 형식 정의
    - role: "user" 또는 "assistant"
    - content: 메시지 내용 문자열
    """
    role: str
    content: str


class ChatRequest(BaseModel):
    """
    /api/chat 엔드포인트에 대한 요청 모델
    - messages: ChatMessage 객체 리스트
    - model: 사용할 LLM 모델명 (기본값: "mistral")
    - stream: 스트리밍 응답 여부 (기본값: False)
    """
    messages: List[ChatMessage]
    model: str = "mistral"
    stream: bool = False


class ChatResponse(BaseModel):
    """
    /api/chat 엔드포인트의 응답 모델
    - response: 생성된 답변 문자열
    - model: 사용된 LLM 모델명
    - created_at: 응답 생성 시각 (옵션)
    """
    response: str
    model: str
    created_at: str


# =============================
# Ollama 관련 설정
# =============================
OLLAMA_URL = "http://localhost:11434"  # Ollama 서버가 실행 중인 URL


# =============================
# RAG 시스템 클래스 정의
# =============================
class RAGSystem:
    """
    RAG(Retrieval-Augmented Generation) 시스템을 담당하는 클래스
    - 데이터 로드
    - 임베딩 생성
    - 쿼리 유사 문서 검색 기능
    """

    def __init__(self, data_folder=None):
        # 초기화 시 데이터 폴더, 문서 리스트, 임베딩, 모델을 준비
        self.data_folder = data_folder or DATA_FOLDER
        self.documents = []  # 문서 저장용 리스트
        self.embeddings = None  # 문서 임베딩 저장용
        self.model = None  # SentenceTransformer 모델 객체
        self.load_model()  # 임베딩 모델 로드
        self.load_documents()  # JSON 문서 로드
        self.create_embeddings()  # 문서 임베딩 생성

    def load_model(self):
        """
        임베딩 모델 로드
        - jhgan/ko-sroberta-multitask 모델을 우선 시도
        - 실패 시 all-MiniLM-L6-v2 모델로 대체
        """
        try:
            logger.info("임베딩 모델 로드 시작...")
            self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            logger.info("임베딩 모델 로드 완료")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            try:
                logger.info("대안 임베딩 모델 로드 시도...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("대안 임베딩 모델 로드 완료")
            except Exception as e2:
                logger.error(f"대안 모델 로드도 실패: {e2}")
                self.model = None

    def load_documents(self):
        """
        JSON 파일들을 로드하여 self.documents 리스트에 문서 단위로 저장
        - data_folder 경로에 있는 모든 .json 파일을 순회하며 load
        - 각 JSON 파일을 process_json_data 메서드로 처리
        """
        logger.info(f"데이터 폴더 확인: {self.data_folder}")

        if not os.path.exists(self.data_folder):
            logger.warning(f"데이터 폴더가 존재하지 않습니다: {self.data_folder}")
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
        """
        JSON 데이터 구조에 맞춰 데이터 처리
        - 크롤링 형식(JSON)에 "menu"와 "content"가 있으면 웹사이트 콘텐츠로 간주
        - 리스트 형식이면 각 아이템을 재귀 호출
        - 그렇지 않으면 legacy 형식으로 처리
        """
        if isinstance(data, dict) and "menu" in data and "content" in data:
            # 웹사이트 크롤링 데이터 처리
            menu = data["menu"]
            content = data["content"]

            # HTML 태그 제거 및 텍스트 정리
            cleaned_content = self.clean_html_content(content)

            # 메뉴별 의미있는 섹션 분할
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
            # 리스트 형태이면 각 아이템을 재귀적으로 처리
            for i, item in enumerate(data):
                if isinstance(item, dict) and "menu" in item and "content" in item:
                    self.process_json_data(item, f"{filename}[{i}]")

        else:
            # legacy JSON 형식 처리
            self.process_legacy_json_data(data, filename)

    def clean_html_content(self, content):
        """
        HTML 태그 제거 및 텍스트 정리
        - BeautifulSoup을 사용해 HTML 파싱
        - 스크립트, 스타일, 네비게이션, 헤더, 푸터 태그 제거
        - 남은 텍스트를 줄바꿈, 공백, 특수문자 정리
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')

            # 불필요한 태그 제거
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()

            # 텍스트 추출
            text = soup.get_text()

            # 텍스트 정리: 연속 줄바꿈, 공백, 특수문자 제거
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s가-힣.,!?()[\]{}:;-]', '', text)

            return text.strip()
        except Exception as e:
            logger.warning(f"HTML 정리 실패: {e}")
            # BeautifulSoup 실패 시 정규식으로 태그 제거
            text = re.sub(r'<[^>]+>', '', content)
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    def extract_meaningful_sections(self, content, menu):
        """
        메뉴에 따라 의미있는 섹션으로 분할
        - 메뉴 이름에 따라 별도 처리 함수 호출 (예: 학사, 식단, 캠퍼스 등)
        - 기본적으로 일반 섹션 분할(extract_general_sections) 호출
        """
        sections = []

        if "업무추진비" in menu:
            sections.extend(self.extract_budget_sections(content, menu))
        elif "학사" in menu or "일정" in menu:
            sections.extend(self.extract_academic_sections(content, menu))
        elif "식단" in menu or "메뉴" in menu:
            sections.extend(self.extract_dining_sections(content, menu))
        elif "캠퍼스" in menu or "찾아오시는" in menu:
            sections.extend(self.extract_campus_sections(content, menu))
        else:
            sections.extend(self.extract_general_sections(content, menu))

        return sections

    def extract_budget_sections(self, content, menu):
        """
        업무추진비 관련 섹션 추출
        - 'YYYY년 MM월 ... 업무추진비 ... 집행 ... 내역' 패턴으로 월별 정보 추출
        - 패턴이 없으면 전체 내용 중 앞 1000자만 섹션으로 추가
        """
        sections = []

        # 월별 업무추진비 내역 추출
        budget_pattern = r'(\d{4}년\s*\d{1,2}월.*?업무추진비.*?집행.*?내역)'
        matches = re.findall(budget_pattern, content, re.IGNORECASE)

        for match in matches:
            sections.append({
                "title": "월별 업무추진비 내역",
                "content": match
            })

        if not sections:
            sections.append({
                "title": "업무추진비 정보",
                "content": content[:1000]
            })

        return sections

    def extract_academic_sections(self, content, menu):
        """
        학사 관련 섹션 추출
        - '개강', '종강', '중간고사', '기말고사' 등 키워드 기반으로 문장 추출
        - 모든 키워드 결과를 합친 뒤, 추가로 전체 앞 1500자 섹션으로도 저장
        """
        sections = []

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

        # 전체 학사 정보 앞 1500자 섹션으로 추가
        sections.append({
            "title": "학사 종합 정보",
            "content": content[:1500]
        })

        return sections

    def extract_dining_sections(self, content, menu):
        """
        식단(메뉴) 관련 섹션 추출
        - 캠퍼스별('공주', '천안', '예산') 키워드와 식사 시간대 키워드('아침', '점심', '저녁')를 조합해 문장 추출
        - 각 캠퍼스별로 추출된 결과를 하나의 섹션으로 저장
        """
        sections = []

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
        """
        캠퍼스 정보 섹션 추출
        - 각 캠퍼스별('공주캠퍼스', '천안캠퍼스', '예산캠퍼스') 주요 키워드 목록을 기반으로 문장 추출
        - 캠퍼스별로 추출된 내용을 하나씩 섹션으로 저장
        """
        sections = []

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
        """
        일반적인 섹션 분할
        - 문장 단위로 split 한 뒤, 500자 단위로 묶어 섹션 생성
        - 각 섹션 제목은 "{menu} 정보"로 통일
        """
        sections = []

        # 문장 단위 분리
        sentences = re.split(r'[.!?]\s+', content)

        current_section = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > 500:
                # 현재까지 모인 문장으로 섹션 저장
                if current_section:
                    sections.append({
                        "title": f"{menu} 정보",
                        "content": '. '.join(current_section) + '.'
                    })
                # 새로운 섹션 초기화
                current_section = [sentence]
                current_length = len(sentence)
            else:
                current_section.append(sentence)
                current_length += len(sentence)

        # 마지막 섹션 남아있으면 추가
        if current_section:
            sections.append({
                "title": f"{menu} 정보",
                "content": '. '.join(current_section) + '.'
            })

        return sections

    def process_legacy_json_data(self, data, filename):
        """
        기존(legacy) JSON 형식 처리
        - 딕셔너리 형태로 key-value 쌍이 있을 때, 콘텐츠와 메타데이터 형태로 documents에 추가
        - 중첩된 딕셔너리가 있으면 키를 "parent.child" 형태로 병합하여 추가
        """
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
        """
        documents 리스트의 모든 문서 내용에 대해 임베딩 생성
        - SentenceTransformer 모델이 준비되어 있어야 함
        - 생성된 임베딩은 self.embeddings에 저장 (shape: [문서 개수, 임베딩 차원])
        """
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
        """
        주어진 쿼리에 대해 문서 간 유사도 계산 후 상위 top_k개 반환
        - 쿼리를 임베딩한 뒤, 기존 문서 임베딩과 코사인 유사도 계산
        - 유사도 0.1 이상인 문서만 결과에 포함
        - 반환 형식: [{"content": ..., "source": ..., "similarity": ...}, ...]
        """
        if self.embeddings is None or not self.model:
            logger.warning("임베딩이나 모델이 없어 검색할 수 없습니다")
            return []

        try:
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query])

            # 문서 임베딩과 코사인 유사도 계산
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]

            # 유사도 상위 인덱스 추출
            top_indices = np.argsort(similarities)[::-1][:top_k]

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


# =============================
# RAG 시스템 초기화 (애플리케이션 실행 시 한 번)
# =============================
logger.info("RAG 시스템 초기화 시작...")
rag_system = RAGSystem()
logger.info("RAG 시스템 초기화 완료")


# =============================
# 기본 엔드포인트 정의
# =============================

@app.get("/")
async def root():
    """
    루트 엔드포인트
    - 서버 상태와 로드된 문서 개수, 디렉터리 정보를 반환
    """
    return {
        "message": "공주대학교 AI 서버가 실행 중입니다",
        "documents_loaded": len(rag_system.documents),
        "data_folder": DATA_FOLDER,
        "log_folder": LOG_FOLDER
    }


@app.get("/health")
async def health_check():
    """
    서버 헬스체크 엔드포인트
    - Ollama 서버 연결 상태 확인 ("/api/tags" 호출)
    - RAG 문서 개수 및 임베딩 생성 여부, 데이터 폴더 존재 여부 반환
    """
    try:
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
    """
    RAG 문서 검색 테스트용 엔드포인트
    - 쿼리 문자열과 top_k 파라미터를 받아 관련 문서를 반환
    """
    try:
        results = rag_system.search_similar_documents(query, top_k)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """
    AI 채팅 엔드포인트 (RAG 적용)
    - 사용자의 마지막 user 메시지를 추출
    - RAG 검색을 통해 관련 문서 3개를 가져와 컨텍스트 생성
    - 시스템 프롬프트에 RAG 컨텍스트 포함하여 Ollama API 호출
    - 응답 결과를 ChatResponse 형태로 반환
    """
    try:
        # 사용자의 마지막 메시지 검색
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        # RAG 검색 수행
        relevant_docs = rag_system.search_similar_documents(user_message, top_k=3)

        # 검색된 문서들을 컨텍스트 문자열로 구성
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

        # 메시지 목록 구성: 시스템 메시지 + 기존 request.messages
        messages = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        # Ollama API 호출용 요청 페이로드 생성
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

        # Ollama 챗 API 호출
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=ollama_request,
            timeout=60
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="AI 서버 오류가 발생했습니다")

        result = response.json()

        # ChatResponse 객체로 응답 데이터 포장
        response_data = ChatResponse(
            response=result["message"]["content"],
            model=request.model,
            created_at=result.get("created_at", "")
        )

        # 디버깅용: 사용된 RAG 문서 로그 출력
        if relevant_docs:
            logger.info(f"RAG 검색 결과: {len(relevant_docs)}개 문서 사용")
            for doc in relevant_docs:
                logger.info(f"- {doc['source']}: {doc['similarity']:.3f}")

        return response_data

    except requests.exceptions.Timeout:
        # 타임아웃 발생 시 504 반환
        raise HTTPException(status_code=504, detail="AI 응답 시간이 초과되었습니다")
    except requests.exceptions.ConnectionError:
        # 연결 오류 시 503 반환
        raise HTTPException(status_code=503, detail="AI 서버에 연결할 수 없습니다")
    except Exception as e:
        logger.error(f"채팅 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


@app.get("/api/models")
async def get_available_models():
    """
    사용 가능한 Ollama 모델 목록 조회 엔드포인트
    - Ollama 서버의 /api/tags 경로 호출
    """
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
    """
    RAG 데이터 다시 로드 엔드포인트
    - 기존 rag_system 인스턴스를 새로 생성하여 JSON 파일을 재로드 및 임베딩 재생성
    """
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


# =============================
# main 함수: uvicorn 서버 실행
# =============================
if __name__ == "__main__":
    # GPU 사용 설정 (CUDA_VISIBLE_DEVICES를 "0"으로 설정하여 첫 번째 GPU만 사용)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    logger.info(f"AI 서버 시작 준비...")
    logger.info(f"프로젝트 루트: {PROJECT_ROOT}")
    logger.info(f"데이터 폴더: {DATA_FOLDER}")
    logger.info(f"로그 폴더: {LOG_FOLDER}")

    # uvicorn으로 FastAPI 앱 실행 (127.0.0.1:8000, workers=1로 설정)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        workers=1,
        log_level="info"
    )
