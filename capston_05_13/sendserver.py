from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from search_utils import build_faiss_index, search_similar_paragraph
from llm_utils import generate_answer_ollama

# 🔐 API 키 설정
API_KEY = "porty-secret-2025"  # 서버 담당자에게 전달할 키

# FastAPI 앱 선언
app = FastAPI()

# CORS 설정 (프론트엔드 또는 외부 접근용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요한 경우 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 바디 정의
class Query(BaseModel):
    question: str

# 문장 데이터 로드
with open("split_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)["results"]
paragraphs = [entry["content"] for entry in data]

# FAISS 인덱스 준비
print("[1] 문장 로드 완료:", len(paragraphs), "개")
index, vectorizer = build_faiss_index(paragraphs)
print("[2] FAISS 인덱스 구축 완료")

# 질문 처리 API
@app.post("/ask")
async def ask(query: Query, x_api_key: str = Header(...)):
    # 🔐 API 키 인증
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="❌ Invalid API Key")

    user_input = query.question

    # 유사 문장 검색
    matched = search_similar_paragraph(user_input, index, vectorizer, paragraphs)
    combined_context = "\n".join(matched[:5])  # 최대 5문장 제한

    # LLM 프롬프트 구성
    prompt = f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.

- 절대로 다른 대학교(예: 국민대학교, 서울대 등)를 언급하거나 생성하지 마세요.  
- 공주대학교는 캠퍼스별로 위치가 분리되어 있으므로, 실제 행정구역을 정확하게 사용하세요.  
- 특히 "공주시 천안동", "공주광역시 천안동" 같은 잘못된 지명은 사용하지 마세요. 
- 천안캠퍼스는 충청남도 천안시, 예산캠퍼스는 충천남도 예산군, 본교는 충청남도 공주시에 위치합니다.


아래의 질문과 관련 문단을 참고하여 공주대학교 기준으로 정확하게 답변하세요.

[질문]
{user_input}

[관련 문단]
{combined_context}

[답변]
"""

    # LLM 호출 (Ollama)
    try:
        answer = generate_answer_ollama(prompt)
    except Exception as e:
        answer = f"⚠️ LLM 응답 오류: {e}"

    return {
        "question": user_input,
        "matched_paragraphs": matched,
        "answer": answer
    }
