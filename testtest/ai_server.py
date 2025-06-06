#ai_server.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import os

from chat_utils import (
    is_small_talk,
    search_similar_paragraphs,
    build_prompt,
    save_log,
)
from embedding_utils import load_embed_model
from llm_utils import generate_answer_ollama
from data_utils import load_paragraphs, prepare_faiss

app = FastAPI(title="공주대학교 AI 서버", version="1.0.0")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "split_results_two.json")
EMBED_PATH = os.path.join(PROJECT_ROOT, "embeddings.npy")
INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss.index")

data = load_paragraphs(DATA_PATH)
tokenizer, model = load_embed_model()
index = prepare_faiss(data, DATA_PATH, EMBED_PATH, INDEX_PATH, tokenizer, model)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    # 1) body.messages에서 마지막 user 메시지를 꺼냅니다.
    user_message = next(
        (m.content for m in reversed(body.messages) if m.role == "user"),
        None
    )
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message provided")

    # 2) 간단한 인삿말 처리
    response = is_small_talk(user_message)
    if response:
        return {"response": response}

    # 3) 유사 문단 검색
    matched_paragraphs = search_similar_paragraphs(
        user_message, data, tokenizer, model, index, top_k=3
    )

    # 4) 프롬프트 작성 → LLM 호출
    prompt = build_prompt(user_message, matched_paragraphs)
    answer = generate_answer_ollama(prompt)

    # 5) 클라이언트 IP 추출
    client_ip = request.client.host

    # 6) 통합 로그 기록
    save_log(
        user_input=user_message,
        matched_paragraphs=matched_paragraphs,
        answer=answer,
        tokenizer=tokenizer,
        model=model,
        client_ip=client_ip
    )

    return {"response": answer}

