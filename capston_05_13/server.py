from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
from nltk.tokenize import sent_tokenize
from search_utils import build_faiss_index, search_similar_paragraph
from embedding_utils import load_embed_model
from llm_utils import generate_answer_ollama

# FastAPI 앱 선언
app = FastAPI()

# 데이터 로드
with open("split_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)["results"]
paragraphs = [entry["content"] for entry in data]

# 임베딩 및 인덱스 준비
print("[1] 문장 로드 완료:", len(paragraphs), "개")
index, vectorizer = build_faiss_index(paragraphs)
print("[2] FAISS 인덱스 구축 완료")

# 입력 형식
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    user_input = query.question

    # 유사 문단 검색
    matched = search_similar_paragraph(user_input, index, vectorizer, paragraphs)
    combined_context = "\n".join(matched)

    # 프롬프트 구성
    prompt = f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.

- 절대로 다른 대학교(예: 국민대학교, 서울대 등)를 언급하거나 생성하지 마세요.  
- 공주대학교는 캠퍼스별로 위치가 분리되어 있으므로, 실제 행정구역을 정확하게 사용하세요.  
- 특히 "공주시 천안동", "공주광역시 천안동" 같은 잘못된 지명은 사용하지 마세요.  
- 천안캠퍼스는 충청남도 천안시, 본교는 충청남도 공주시에 위치합니다.

아래의 질문과 관련 문단을 참고하여 공주대학교 기준으로 정확하게 답변하세요.

[질문]
{user_input}

[관련 문단]
{combined_context}

[답변]
"""

    # LLM 호출
    answer = generate_answer_ollama(prompt)

    return {
        "question": user_input,
        "matched_paragraphs": matched,
        "answer": answer
    }
