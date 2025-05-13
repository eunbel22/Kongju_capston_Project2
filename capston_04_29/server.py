# server.py
#서버랑 연결해서 테스트용

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding_utils import load_embed_model
from search_utils import build_faiss_index, search_similar_paragraph
from llm_utils import generate_answer_ollama
from text_utils import extract_keywords
from data_loader import load_paragraphs_and_embeddings
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

# 🔄 정제된 JSON 사용
with open("kongju_cleaned.json", "r", encoding="utf-8") as f:
    kongju_data = json.load(f)

all_text = " ".join([entry['content'] for entry in kongju_data])
keywords = extract_keywords(all_text)
paragraphs, paragraph_embeddings = load_paragraphs_and_embeddings()
index = build_faiss_index(paragraph_embeddings)
embed_tokenizer, embed_model = load_embed_model()

@app.post("/question")
def ask_question(req: QuestionRequest):
    try:
        matched_paragraph = search_similar_paragraph(
            req.question, index, paragraphs, embed_tokenizer, embed_model, keywords
        )

        prompt = f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.

        - 절대로 다른 대학교(예: 국민대학교, 서울대 등)를 언급하거나 생성하지 마세요.  
        - 공주대학교는 캠퍼스별로 위치가 분리되어 있으므로, 실제 행정구역을 정확하게 사용하세요.  
        - 특히 "공주시 천안동", "공주광역시 천안동" 같은 잘못된 지명은 사용하지 마세요.  
        - 천안캠퍼스는 충청남도 천안시, 본교는 충청남도 공주시에 위치합니다.

        아래의 질문과 관련 문단을 참고하여 공주대학교 기준으로 정확하게 답변하세요.

        [질문]
        {req.question}

        [관련 문단]
        {matched_paragraph}

        [답변]
        """

        answer = generate_answer_ollama(prompt)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




#가상 환경으로 이동 후 cd를 사용해서 server.py파일이 있는 곳으로  이동후 uvicorn server:app --reload를 입력하면 됨
