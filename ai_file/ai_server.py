import os
import json
import re
import numpy as np
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List

from chat_utils import (
    is_small_talk,
    is_casual_or_emotional, 
    build_prompt,
    save_log,
    extract_keywords
)
from embedding_utils import load_embed_model, embed_texts
from llm_utils import generate_answer_qwen, load_llm  # ← 수정: generate_answer_qwen 추가
from data_utils import load_paragraphs, prepare_faiss
from config import (
    DATA_PATH,
    SHUTTLE_PATH,
    EMBED_PATH,
    INDEX_PATH,
    PROFANITY_PATH,
    CRAWL_BASE_URL,
    ENVIRONMENT
)

app = FastAPI(title="공주대학교 AI 서버", version="1.0.0", debug=True)

print(f"[서버 모드] {ENVIRONMENT}")

# shuttlebus.json 로드
try:
    with open(SHUTTLE_PATH, "r", encoding="utf-8") as f:
        shuttle_data = json.load(f)
except:
    shuttle_data = None

# ── 고정 응답 상수 ─────────────────────────────────────────────────────────────
CANNED_RESPONSE = (
    "포티는 공주대학교에 대한 정보만 알고 있어요💡 "
    "학교와 관련된 질문이라면 뭐든 도와드릴게요! "
    "도움이 필요하시면 [도움말]을 입력해주세요"
)
HELP_RESPONSE = (
    "도움이 필요한 경우 다음과 같은 형식으로 질문해주시면 됩니다:\n"
    "1. 캠퍼스 정보\n"
    "- 천안캠퍼스 위치 알려줘\n"
    "2. 학사 일정\n"
    "- 2025년 4월 학사일정 알려줘\n"
    "- 개강일이 언제야?\n"
    "3. 도서관/식당 정보\n"
    "- 도서관 운영시간이 몇 시야?\n"
    "- 천안 식단이 뭐야?\n"
    "4. 전공·교과 관련\n"
    "- 소프트웨어학과에서는 뭐 배워?\n"
    "궁금한 점이 있으면, 위 예시를 참고하셔서 자유롭게 질문해주세요.\n"
    "※ 공주대학교와 관계없는 질문은 정확히 답변하기 어려워요.\n"
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
print("[초기화] 데이터 로딩 중...")
data = load_paragraphs(DATA_PATH)
tokenizer, model = load_embed_model()
index = prepare_faiss(data, DATA_PATH, EMBED_PATH, INDEX_PATH, tokenizer, model)
print("[초기화] RAG 시스템 준비 완료!")

# ── LLM 모델 로드 (새로 추가!) ────────────────────────────────────────────────
print("[초기화] LLM 모델 로딩 중...")
llm_tokenizer, llm_model = load_llm()  # ← 추가: 서버 시작 시 LLM 사전 로드
print("[초기화] LLM 모델 로드 완료!")


# ── Request/Response 모델 ───────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


def get_client_ip(request: Request) -> str:
    """실제 클라이언트 IP를 추출하는 함수"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()
    
    forwarded_host = request.headers.get("X-Forwarded-Host")
    if forwarded_host:
        return forwarded_host.strip()
    
    return request.client.host if request.client else "unknown"


# ── API 엔드포인트 ─────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    # 1) 마지막 사용자 메시지 추출
    user_message = next((m.content for m in reversed(body.messages) if m.role == "user"), None)
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

    # ── 셔틀/버스 질문 처리 ────────────────────────────────────────────────
    if any(kw in user_message for kw in ("셔틀", "버스")) and shuttle_data:
        locations = re.findall(r"[가-힣]+", user_message)
        
        routes = [
            r for r in shuttle_data["shuttle_schedules"]
            if any(loc in r["route"] for loc in locations)
        ]
        if not routes:
            routes = [shuttle_data["shuttle_schedules"][0]]

        context = {
            "service_period": shuttle_data["service_period"],
            "routes": routes
        }
        shuttle_ctx = json.dumps(context, ensure_ascii=False)

        prompt = f"""다음은 공주대학교 셔틀버스 운영 정보입니다:
{shuttle_ctx}

위 정보를 참고해 아래 질문에 답해주세요.

[질문]
{user_message}

[답변]
"""
        # ← 수정: llm_tokenizer, llm_model 전달
        answer = generate_answer_qwen(prompt, llm_tokenizer, llm_model)
        
        client_ip = request.client.host
        save_log(
            user_input=user_message,
            matched_paragraphs=[],
            answer=answer,
            tokenizer=tokenizer,
            model=model,
            client_ip=client_ip
        )
        return {"response": answer}

    # "공주대학교에 대해 알려줘" 요청 처리
    if "공주대학교에 대해 알려줘" in user_message:
        intro = (
            "안녕하세요! 공주대학교에 대해 알려드릴게요 😊\n\n"
            "• 설립 연도 및 유형\n"
            "  – 1948년 '공주사범학교'로 개교\n"
            "  – 1979년 종합대학으로 승격된 국립 종합대학교\n\n"
            "• 캠퍼스 구성\n"
            "  – 공주캠퍼스(본교): 인문·사회·자연과학 계열\n"
            "  – 천안캠퍼스: 공과대학(기계·전기·화학공학 등)\n"
            "  – 예산캠퍼스: 생명과학·산림자원 분야\n\n"
            "더 궁금하신 점이 있으면 언제든 질문해주세요!"
        )
        return {"response": intro}

    # 메뉴/식단 질문 처리
    if any(k in user_message for k in ("메뉴", "식단", "식당", "점심", "저녁", "아침")):
        campus = "cheonan"
        mapping = {
            "천안": "cheonan",
            "드림": "dream",
            "예산": "yesan",
            "은행사": "ehh",
            "홍익사": "ehh",
            "해오름": "ehh",
            "비전": "VB",
            "블룸": "VB",
        }
        for kr, en in mapping.items():
            if kr in user_message:
                campus = en
                break
        else:
            for code in mapping.values():
                if code in user_message.lower():
                    campus = code
                    break

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{CRAWL_BASE_URL}/crawl/{campus}", timeout=30.0)
                resp.raise_for_status()
                crawl_data = resp.json()
            except Exception as e:
                return {"response": f"실시간 식단 정보 조회 중 오류 발생: {e}"}

        today = datetime.now().strftime("%m월 %d일")
        today_meal = next(
            (m for m in crawl_data.get("meal", [])
               if today in m.get("date","").strip()),
            None
        )

        if not today_meal:
            return {"response": f"{campus.capitalize()} 캠퍼스 식단 정보가 준비되지 않았습니다."}

        def clean(item: str) -> str:
            if not item:
                return ''
            return re.sub(r"\s*\d+\s?kcal|\s*\d+\s?g", '', item)

        bf = clean(today_meal.get('breakfast'))
        lf = clean(today_meal.get('lunch'))
        dn = clean(today_meal.get('dinner'))

        meal_type = None
        if "아침" in user_message:
            meal_type = "아침"
        elif "점심" in user_message:
            meal_type = "점심"
        elif "저녁" in user_message:
            meal_type = "저녁"

        if meal_type == "아침":
            answer = f"{campus.capitalize()}캠퍼스 오늘({today}) 아침 메뉴입니다:\n☀ {bf or '정보 없음'}"
        elif meal_type == "점심":
            answer = f"{campus.capitalize()}캠퍼스 오늘({today}) 점심 메뉴입니다:\n🌤 {lf or '정보 없음'}"
        elif meal_type == "저녁":
            answer = f"{campus.capitalize()}캠퍼스 오늘({today}) 저녁 메뉴입니다:\n🌙 {dn or '정보 없음'}"
        else:
            answer = (
                f"{campus.capitalize()} 캠퍼스 오늘({today}) 식단입니다:\n\n"
                f"☀ 아침: {bf or '정보 없음'}\n"
                f"🌤 점심: {lf or '정보 없음'}\n"
                f"🌙 저녁: {dn or '정보 없음'}"
            )

        save_log(
            user_input=user_message,
            matched_paragraphs=[],
            answer=answer,
            tokenizer=tokenizer,
            model=model,
            client_ip=request.client.host
        )
        return {"response": answer}

    # 5) 쿼리 임베딩 계산
    q_emb = embed_texts([user_message], tokenizer, model)[0]

    # 6) FAISS 검색 (상위 3개)
    _, idxs = index.search(q_emb.reshape(1, -1), 3)
    matched = [data[i] for i in idxs[0] if i < len(data)]

    # 7) 핵심 키워드 추출 및 문단 필터링
    keywords = extract_keywords(user_message)
    if keywords:
        matched = [
            p for p in matched
            if any(kw in p.get("content", "") for kw in keywords)
        ]

    # 8) 코사인 유사도 계산 및 임계값 검사
    sims = []
    for p in matched:
        p_emb = embed_texts([p.get("content", "")], tokenizer, model)[0]
        sim = np.dot(q_emb, p_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(p_emb) + 1e-9)
        sims.append(sim)
    max_sim = max(sims) if sims else 0.0

    # 9) RAG 성공: 공주대 정보 제공
    if matched and max_sim >= 0.45:
        prompt = build_prompt(user_message, matched)
        answer = generate_answer_qwen(prompt, llm_tokenizer, llm_model)

        client_ip = get_client_ip(request)
        save_log(
            user_input=user_message,
            matched_paragraphs=matched,
            answer=answer,
            tokenizer=tokenizer,
            model=model,
            client_ip=client_ip
        )
        return {"response": answer}

    # ========== 🆕 10) 하이브리드: 일상 대화 LLM 처리 ==========
    if is_casual_or_emotional(user_message):
        casual_prompt = f"""당신은 공주대학교 정보 챗봇 포티입니다.

사용자가 공주대와 직접 관련 없는 일상적 이야기를 했습니다.
하지만 친절하게 공감하고, 자연스럽게 공주대 관련 정보로 연결하세요.

[사용자 입력]
{user_message}

[응답 지침]
1. 사용자 감정/상황에 공감 표현
2. 공주대 관련 서비스나 정보로 자연스럽게 유도
3. 구체적인 질문 유도

[응답 예시]
- "배고파" → "배고프시군요! 🍚 공주대 식당 메뉴 확인해보실래요? '천안캠퍼스 오늘 식단'처럼 물어보세요!"
- "힘들어" → "힘드시군요 😔 학교 생활이 버거울 땐 학생상담센터(☎ 041-850-8197)를 이용해보세요!"
- "심심해" → "그럴 땐 동아리 활동이 어때요? 관심 분야를 말씀해주시면 추천해드릴게요!"
- "뭐하면 좋을까" → "공주대에는 다양한 활동이 있어요! 동아리, 학술대회, 행사 등 어떤 게 궁금하신가요?"

[답변]
"""
        answer = generate_answer_qwen(casual_prompt, llm_tokenizer, llm_model)
        
        client_ip = get_client_ip(request)
        save_log(
            user_input=user_message,
            matched_paragraphs=[],  # RAG 미사용
            answer=answer,
            tokenizer=tokenizer,
            model=model,
            client_ip=client_ip
        )
        return {"response": answer}

    # 11) 완전히 무관한 질문
    return {"response": CANNED_RESPONSE}