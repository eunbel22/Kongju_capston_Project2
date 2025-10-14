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
    build_prompt,
    save_log,
    extract_keywords  # ← 추가
)
from embedding_utils import load_embed_model, embed_texts
from llm_utils import generate_answer_ollama
from data_utils import load_paragraphs, prepare_faiss

app = FastAPI(title="공주대학교 AI 서버", version="1.0.0", debug=True)

# ── 파일 경로 설정 ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "split_results_two.json")
SHUTTLE_PATH   = os.path.join(PROJECT_ROOT, "shuttlebus.json")
EMBED_PATH = os.path.join(PROJECT_ROOT, "embeddings.npy")
INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss.index")
PROFANITY_PATH = os.path.join(PROJECT_ROOT, "profanity.json")


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
    "공주대학교와 관련이 없는 질문이나 정보가 없는 글은 “포티는 공주대학교에 대한 정보만 알고 있어요💡\n"
    "학교와 관련된 질문이라면 뭐든 도와드릴게요!\n"
    "도움이 필요하시면 [도움말]을 입력해주세요”라고 대답합니다."
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
data = load_paragraphs(DATA_PATH)
tokenizer, model = load_embed_model()
index = prepare_faiss(data, DATA_PATH, EMBED_PATH, INDEX_PATH, tokenizer, model)


# ── Request/Response 모델 ───────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


CRAWL_BASE_URL = "http://127.0.0.1:8001"  # uvicorn realtime_crawling:app 으로 띄운 서버


def get_client_ip(request: Request) -> str:
    """
    실제 클라이언트 IP를 추출하는 함수
    프록시나 로드밸런서를 통해 들어오는 요청도 처리
    """
    # X-Forwarded-For 헤더 확인 (가장 일반적)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # 여러 IP가 있을 경우 첫 번째가 실제 클라이언트 IP
        return forwarded_for.split(",")[0].strip()

    # X-Real-IP 헤더 확인
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # CF-Connecting-IP 헤더 확인 (Cloudflare)
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # X-Forwarded-Host 헤더 확인
    forwarded_host = request.headers.get("X-Forwarded-Host")
    if forwarded_host:
        return forwarded_host.strip()

    # 기본값: request.client.host
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
    # shuttlebus.json 에 있는 데이터를 컨텍스트로 LLM 호출

    if any(kw in user_message for kw in ("셔틀", "버스")) and shuttle_data:
        # 1) user_message 에서 한글 지명(예: '대전', '예산') 추출
        locations = re.findall(r"[가-힣]+", user_message)

        # 2) shuttle_data 중 관련 노선만 필터
        routes = [
            r for r in shuttle_data["shuttle_schedules"]
            if any(loc in r["route"] for loc in locations)
        ]
        # 매칭된 노선이 없으면 기본 천안 노선 한 개만
        if not routes:
            routes = [shuttle_data["shuttle_schedules"][0]]

        # 3) 서비스 기간 + 매칭된 노선만 담은 컨텍스트로 직렬화
        context = {
            "service_period": shuttle_data["service_period"],
            "routes": routes
        }
        shuttle_ctx = json.dumps(context, ensure_ascii=False)

        prompt = f"""
    다음은 공주대학교 셔틀버스 운영 정보입니다(운행 기간, 노선, 정류장, 시간표 전체 JSON):
    {shuttle_ctx}
    
    위 JSON 정보를 참고해, 아래 질문에 알맞게 답해주세요.
    
    [질문]
    {user_message}
    
    [답변]
    """

        answer = generate_answer_ollama(prompt)

        # ── 셔틀 LLM 호출도 save_log로 남기기 ───────────────────────────
        client_ip = request.client.host
        save_log(
            user_input = user_message,
            matched_paragraphs = [],  # RAG가 아니므로 빈 리스트
            answer = answer,
            tokenizer = tokenizer,
            model = model,
            client_ip = client_ip
        )
        # ──────────────────────────────────────────────────────────────

        return {"response": answer}
    # ─────────────────────────────────────────────────────────────────

    

    # “공주대학교에 대해 알려줘” 요청 처리
    if "공주대학교에 대해 알려줘" in user_message:
        intro = (
            "안녕하세요! 공주대학교에 대해 알려드릴게요 😊\n\n"
            "• 설립 연도 및 유형\n"
            "  – 1948년 ‘공주사범학교’로 개교\n"
            "  – 1979년 종합대학으로 승격된 국립 종합대학교\n\n"
            "• 캠퍼스 구성\n"
            "  – 공주캠퍼스(본교): 인문·사회·자연과학 계열\n"
            "  – 천안캠퍼스: 공과대학(기계·전기·화학공학 등)\n"
            "  – 예산캠퍼스: 생명과학·산림자원 분야\n\n"
            "• 주요 학부·대학원\n"
            "  – 인문사회과학대학, 자연과학대학, 공과대학, 사범대학, 예술대학 등\n"
            "  – 석·박사 통합과정 및 특수대학원 운영\n\n"
            "• 연구역량 및 시설\n"
            "  – 국책 과제 수행, LINC+ 산학협력사업 참여\n"
            "  – 첨단 실험·연구시설(공동실험관, 첨단세라믹연구소 등)\n\n"
            "• 학생·교류 프로그램\n"
            "  – 약 1만 5천명 재학생, 다국적 교환학생(30여개국)\n"
            "  – 창업보육센터·취창업지원센터 운영\n\n"
            "• 캠퍼스 라이프\n"
            "  – 중앙도서관, 학생식당, 기숙사, 동아리, 체육관 등\n\n"
            "더 궁금하신 점이 있으면 언제든 질문해주세요!"
        )
        return {"response": intro}




      # 메뉴/식단 질문일 때
    if any(k in user_message for k in ("메뉴", "식단", "식당", "점심", "저녁", "아침")):
        # 캠퍼스 결정
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
        # 먼저 한글 키워드로 체크
        for kr, en in mapping.items():
            if kr in user_message:
                campus = en
                break
        else:
            # 한글 키워드가 없으면 영어 코드로 체크
            for code in mapping.values():
                if code in user_message.lower():
                    campus = code
                    break


        # 디버그: 캠퍼스가 뭘로 잡혔는지 터미널에 출력
        print(f"[DEBUG] 메뉴/식단 감지 → campus='{campus}', user_message='{user_message}'")


        # 크롤러 호출
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    f"{CRAWL_BASE_URL}/crawl/{campus}",
                    timeout=30.0  # read/connect 모두 30초로 연장
                )
                resp.raise_for_status()
                crawl_data = resp.json()
                print(f"[DEBUG] 크롤러 응답 데이터 크기: {len(crawl_data.get('meal', []))}")

            except Exception as e:
                # 예외 종류와 메시지를 모두 찍어 줍니다
                print("[ERROR] 크롤러 호출 중 예외 발생!", repr(e))
                return {"response": f"실시간 식단 정보 조회 중 오류 발생: {e}"}

        # 오늘 날짜 문자열
        today = datetime.now().strftime("%m월 %d일")

        # 오늘 식단 찾기
        today_meal = next(
            (m for m in crawl_data.get("meal", [])
               if today in m.get("date","").strip()),
            None
        )

        if not today_meal:
            return {"response": f"{campus.capitalize()} 캠퍼스 식단 정보가 준비되지 않았습니다."}

        #영양정보 제거 함수
        def clean(item: str) -> str:
            if not item:
                return ''
            return re.sub(r"\s*\d+\s?kcal|\s*\d+\s?g", '', item)

        bf = clean(today_meal.get('breakfast'))
        lf = clean(today_meal.get('lunch'))
        dn = clean(today_meal.get('dinner'))

        # ————— 식사 타입 결정 —————
        meal_type = None
        if "아침" in user_message:
            meal_type = "아침"
        elif "점심" in user_message:
            meal_type = "점심"
        elif "저녁" in user_message:
            meal_type = "저녁"

        # ————— 답변 생성 —————
        if meal_type == "아침":
            answer = f"{campus.capitalize()}캠퍼스 오늘({today}) 아침 메뉴입니다:\n☀ {bf or '정보 없음'}"
        elif meal_type == "점심":
            answer = f"{campus.capitalize()}캠퍼스 오늘({today}) 점심 메뉴입니다:\n🌤 {lf or '정보 없음'}"
        elif meal_type == "저녁":
            answer = f"{campus.capitalize()}캠퍼스 오늘({today}) 저녁 메뉴입니다:\n🌙 {dn or '정보 없음'}"
        else:
            # 기존처럼 전부 보여주려면 이 블록 유지
            answer = (
                f"{campus.capitalize()} 캠퍼스 오늘({today}) 식단입니다:\n\n"
                f"☀ 아침: {bf or '정보 없음'}\n"
                f"🌤 점심: {lf or '정보 없음'}\n"
                f"🌙 저녁: {dn or '정보 없음'}"
            )

        # [22] 식단 조회 로그 저장 및 응답에 문단 포함
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

    # 9) 매칭 실패 혹은 유사도 낮음 → 고정 응답
    if not matched or max_sim < 0.45:  # ← threshold를 0.45로 상향 조정
        return {"response": CANNED_RESPONSE}

    # 10) 프롬프트 작성 → LLM 호출
    prompt = build_prompt(user_message, matched)
    answer = generate_answer_ollama(prompt)

    # 11) 클라이언트 IP 추출 (개선된 방식)
    client_ip = get_client_ip(request)
    print(f"[DEBUG] 클라이언트 IP: {client_ip}")

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