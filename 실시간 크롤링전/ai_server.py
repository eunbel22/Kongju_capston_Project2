# 📁 ai_server.py

import os
import json
import re
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List

from chat_utils import (
    is_small_talk,
    build_prompt,
    save_log,
    extract_keywords     # ← 추가
)
from embedding_utils import load_embed_model, embed_texts
from llm_utils import generate_answer_ollama
from data_utils import load_paragraphs, prepare_faiss

app = FastAPI(title="공주대학교 AI 서버", version="1.0.0")

# ── 파일 경로 설정 ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH      = os.path.join(PROJECT_ROOT, "split_results_two.json")
EMBED_PATH     = os.path.join(PROJECT_ROOT, "embeddings.npy")
INDEX_PATH     = os.path.join(PROJECT_ROOT, "faiss.index")
PROFANITY_PATH = os.path.join(PROJECT_ROOT, "profanity.json")

# ── 고정 응답 상수 ─────────────────────────────────────────────────────────────
CANNED_RESPONSE = (
    "포티는 공주대학교에 대한 정보만 알고 있어요💡 "
    "학교와 관련된 질문이라면 뭐든 도와드릴게요! "
    "도움이 필요하시면 [도움말]을 입력해주세요"
)
HELP_RESPONSE = (
    "안녕하세요! 저는 공주대학교 AI 챗봇 포티입니다 😊\n\n"
    "도움이 필요한 경우 다음과 같은 형식으로 질문해주시면 됩니다:\n"
    "1. 캠퍼스 정보\n"
    "2. 학사 일정\n"
    "3. 도서관/식당 정보\n"
    "4. 전공·교과 관련\n"
    "5. 기타\n"
    "※ 공주대학교와 관계없는 질문은 정확히 답변하기 어려워요."
)

# ── 비속어 목록 로드 ───────────────────────────────────────────────────────────
try:
    with open(PROFANITY_PATH, "r", encoding="utf-8") as f:
        PROFANITY_LIST = json.load(f).get("bad_words", [])
except Exception:
    PROFANITY_LIST = []

def contains_profanity(text: str) -> bool:
    t = text.lower()
    for bad in PROFANITY_LIST:
        if re.search(rf"\b{re.escape(bad)}\b", t):
            return True
    return False

# ── RAG 준비 ──────────────────────────────────────────────────────────────────
data      = load_paragraphs(DATA_PATH)
tokenizer, model = load_embed_model()
index     = prepare_faiss(data, DATA_PATH, EMBED_PATH, INDEX_PATH, tokenizer, model)

# ── Request/Response 모델 ───────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# ── API 엔드포인트 ─────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    # 1) 마지막 사용자 메시지 추출
    user_message = next((m.content for m in reversed(body.messages) if m.role=="user"), None)
    if not user_message:
        raise HTTPException(400, "No user message provided")

    # 2) 비속어 감지
    if contains_profanity(user_message):
        return {"response": CANNED_RESPONSE}

    # 3) 도움말 요청 처리
    if user_message.strip() in ("도와줘", "도움말"):
        return {"response": HELP_RESPONSE}

    # 4) 간단 인삿말 처리
    small = is_small_talk(user_message)
    if small:
        return {"response": small}

    # 5) 쿼리 임베딩 계산
    q_emb = embed_texts([user_message], tokenizer, model)[0]

    # 6) FAISS 검색 (상위 3개)
    _, idxs = index.search(q_emb.reshape(1,-1), 3)
    matched = [data[i] for i in idxs[0] if i < len(data)]

    # 7) 핵심 키워드 추출 및 문단 필터링
    keywords = extract_keywords(user_message)
    if keywords:
        matched = [
            p for p in matched
            if any(kw in p.get("content","") for kw in keywords)
        ]

    # 8) 코사인 유사도 계산 및 임계값 검사
    sims = []
    for p in matched:
        p_emb = embed_texts([p.get("content","")], tokenizer, model)[0]
        sim = np.dot(q_emb, p_emb) / (np.linalg.norm(q_emb)*np.linalg.norm(p_emb) + 1e-9)
        sims.append(sim)
    max_sim = max(sims) if sims else 0.0

    # 9) 매칭 실패 혹은 유사도 낮음 → 고정 응답
    if not matched or max_sim < 0.45:  # ← threshold를 0.45로 상향 조정
        return {"response": CANNED_RESPONSE}

    # 10) 프롬프트 작성 → LLM 호출
    prompt = build_prompt(user_message, matched)
    answer = generate_answer_ollama(prompt)

    # 11) 클라이언트 IP 추출
    client_ip = request.client.host

    # 12) 로그 기록
    save_log(
        user_input=user_message,
        matched_paragraphs=matched,
        answer=answer,
        tokenizer=tokenizer,
        model=model,
        client_ip=client_ip
    )

    # 13) 최종 응답
    return {"response": answer}
