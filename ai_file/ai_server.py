from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import json
from chat_utils import is_small_talk, search_similar_paragraphs, build_prompt, save_log
from embedding_utils import load_embed_model
from llm_utils import generate_answer_ollama
from data_utils import load_paragraphs, prepare_faiss

# FastAPI 앱 생성
app = FastAPI(title="공주대학교 AI 서버", version="1.0.0")

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "split_results_two.json")
EMBED_PATH = os.path.join(PROJECT_ROOT, "embeddings.npy")
INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss.index")

# 데이터 및 모델 로딩
data = load_paragraphs(DATA_PATH)
tokenizer, model = load_embed_model()
index = prepare_faiss(data, EMBED_PATH, INDEX_PATH, tokenizer, model)

# 요청 모델 정의
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message provided")

    # 인삿말 분기 처리
    response = is_small_talk(user_message)
    if response:
        return {"response": response}

    # 유사 문단 검색
    matched_paragraphs = search_similar_paragraphs(user_message, data, tokenizer, model, index, top_k=3)
    prompt = build_prompt(user_message, matched_paragraphs)

    # LLM 응답 생성
    answer = generate_answer_ollama(prompt)

    # 로그 저장
    save_log(user_message, answer)

    return {"response": answer}
