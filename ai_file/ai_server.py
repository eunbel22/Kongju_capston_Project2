#ai_server.py
import os
import json
import re
import numpy as np
from datetime import datetime, timedelta
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from collections import defaultdict
import torch
from config import USE_MILVUS, MILVUS_CONFIG

from chat_utils import (
    is_small_talk,
    build_prompt,
    save_log,
    extract_keywords,
    expand_pronouns_with_history,
    is_casual_chat,      # 🚀 하이브리드: 일상 대화 감지
    build_casual_prompt  
)
from embedding_utils import load_embed_model, embed_texts
from llm_utils import generate_answer_qwen, load_llm
from data_utils import load_paragraphs, prepare_faiss
from profanity_filter import contains_profanity
from config import (
    DATA_FILES,
    SHUTTLE_PATH,
    EMBED_PATH,
    INDEX_PATH,
    PROFANITY_PATH,
    CRAWL_BASE_URL,
    ENVIRONMENT
)

# ai_server.py 상단 (35번 라인 근처)

if USE_MILVUS:
    from milvus_utils import get_milvus_client
    print("[초기화] Milvus 모드 활성화")
    try:
        milvus_client = get_milvus_client()
        if milvus_client is None:
            print("[경고] Milvus 연결 실패, FAISS로 폴백")
            USE_MILVUS = False  # ← 자동으로 FAISS 모드로 전환
    except Exception as e:
        print(f"[경고] Milvus 초기화 실패: {e}")
        print("[경고] FAISS 모드로 폴백")
        USE_MILVUS = False
else:
    print("[초기화] FAISS 모드 (기존)")

app = FastAPI(title="공주대학교 AI 서버", version="1.0.0", debug=True)

print(f"[서버 모드] {ENVIRONMENT}")

# ========== 대화 히스토리 저장소 ==========
conversation_history = defaultdict(list)
session_last_activity = {}

SESSION_TIMEOUT = timedelta(minutes=30)

def cleanup_old_sessions():
    """30분 이상 비활성 세션 삭제"""
    now = datetime.now()
    expired = [
        sid for sid, last_time in session_last_activity.items()
        if now - last_time > SESSION_TIMEOUT
    ]
    for sid in expired:
        del conversation_history[sid]
        del session_last_activity[sid]
    
    if expired:
        print(f"[세션 정리] {len(expired)}개 세션 삭제")

def get_session_id(request: Request) -> str:
    """클라이언트 IP 기반 세션 ID 생성"""
    client_ip = get_client_ip(request)
    return f"session_{client_ip}"

def add_to_history(session_id: str, role: str, content: str):
    """대화 히스토리에 추가"""
    conversation_history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })
    
    if len(conversation_history[session_id]) > 20:
        conversation_history[session_id] = conversation_history[session_id][-20:]
    
    session_last_activity[session_id] = datetime.now()

def get_history(session_id: str, max_turns: int = 3) -> List[dict]:
    """최근 N턴의 대화 가져오기"""
    history = conversation_history.get(session_id, [])
    return history[-(max_turns * 2):]

# shuttlebus.json 로드
try:
    with open(SHUTTLE_PATH, "r", encoding="utf-8") as f:
        shuttle_data = json.load(f)
except:
    shuttle_data = None

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

print("[초기화] 데이터 로딩 중...")

# 여러 파일 병합
data = []
for path in DATA_FILES:
    paragraphs = load_paragraphs(path)
    data.extend(paragraphs)

tokenizer, model = load_embed_model()

# 가장 최근 파일 기준으로 임베딩 업데이트
main_json_path = DATA_FILES[0]

index = prepare_faiss(
    data,
    main_json_path,   # json 최신 수정 시간 체크용
    EMBED_PATH,
    INDEX_PATH,
    tokenizer,
    model
)

print("[초기화] RAG 시스템 준비 완료!")

print("[초기화] LLM 모델 로딩 중...")
llm_tokenizer, llm_model = load_llm()
print("[초기화] LLM 모델 로드 완료!")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    sessionId: Optional[str] = None
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


# ========== ✅ is_casual_conversation 함수 삭제 (is_casual_chat 사용) ==========


@app.get("/api/ai/health")
async def ai_health():
    """AI 서버 헬스 체크"""
    return {"status": "ok"}

@app.post("/api/ai/query")
async def chat(request: Request, body: ChatRequest):
    cleanup_old_sessions()
    session_id = get_session_id(request)
    
    user_message = next((m.content for m in reversed(body.messages) if m.role == "user"), None)
    if not user_message:
        raise HTTPException(400, "No user message provided")

    add_to_history(session_id, "user", user_message)

    # ========== 1. 비속어 체크 ==========
    if contains_profanity(user_message):
        print(f"[비속어 차단] 세션: {session_id}, 입력: '{user_message}'")
        answer = CANNED_RESPONSE
        add_to_history(session_id, "assistant", answer)
        return {"response": answer}

    # ========== 2. 도움말 ==========
    if user_message.strip() in ("도와줘", "도움말"):
        answer = HELP_RESPONSE
        add_to_history(session_id, "assistant", answer)
        return {"response": answer}

    # ========== 3. Small Talk ==========
    small = is_small_talk(user_message)
    if small:
        add_to_history(session_id, "assistant", small)
        return {"response": small}
    
    

    # ========== ✅ 4. 대명사 확장 (가장 먼저!) ==========
    history = get_history(session_id, max_turns=3)
    expanded_message = expand_pronouns_with_history(user_message, history)
    
    # 디버그 로그
    if expanded_message != user_message:
        print(f"[대명사 확장] ✅ 성공!")
        print(f"  원본: {user_message}")
        print(f"  확장: {expanded_message}")
    
    # ✅ 이후부터는 expanded_message 사용!
    user_message = expanded_message

    # ========== 🚀 5. 일상 대화 (is_casual_chat 사용!) ==========
    if is_casual_chat(user_message):  # ← is_casual_conversation에서 is_casual_chat으로 변경!
        print(f"[일상 대화] 감지: {user_message}")
        
        system_content = "당신은 공주대 학생들의 친구 포티입니다. 이전 대화를 기억하며 자연스럽게 대화하세요. 반말로 1-2문장만 짧게 답변하세요."
        
        messages = [{"role": "system", "content": system_content}]
        
        for msg in history[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            prompt = llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            prompt = f"{system_content}\n\n"
            for msg in messages[1:]:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant: "
        
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
        
        if ENVIRONMENT == 'production':
            max_tokens, temp = 80, 0.6
        else:
            max_tokens, temp = 50, 0.7
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
                repetition_penalty=1.3,
                early_stopping=True
            )
        
        answer = llm_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        if ENVIRONMENT != 'production':
            for sep in ['. ', '! ', '? ', '\n']:
                if sep in answer:
                    answer = answer.split(sep)[0] + sep.strip()
                    break
            if len(answer) > 50:
                answer = answer[:47] + '...'
        
        if not answer or len(answer) < 3:
            answer = "공주대 얘기면 뭐든 물어봐! 😊"
        
        add_to_history(session_id, "assistant", answer)
        
        client_ip = get_client_ip(request)
        save_log(user_input=user_message, matched_paragraphs=[], answer=answer,
                 tokenizer=tokenizer, model=model, client_ip=client_ip)
        return {"response": answer}

    # ========== ✅ 6. 셔틀버스 (확장된 메시지 사용) ==========
    if any(kw in user_message for kw in ("셔틀", "버스")) and shuttle_data:
        print(f"[셔틀버스] 처리 시작: {user_message}")
        
        # ✅ 출발지와 목적지를 구분해서 추출
        departure = None
        destination = None
        
        # 캠퍼스 매핑
        campus_map = {
            "천안캠퍼스": "천안", "천안공과대학": "천안", "천안": "천안",
            "공주캠퍼스": "공주", "공주": "공주", "본교": "공주",
            "예산캠퍼스": "예산", "예산": "예산", "산업과학대학": "예산",
            "소프트웨어공학과": "천안", "기계공학과": "천안", "전기전자공학과": "천안",
            "대전": "대전", "청주": "청주", "세종": "세종", "유성": "유성"
        }
        
        # "에서" 앞 = 출발지
        from_match = re.search(r"([가-힣]+)(에서|부터)", user_message)
        if from_match:
            place = from_match.group(1)
            for key, value in campus_map.items():
                if key in place:
                    departure = value
                    print(f"[셔틀버스] 출발지 발견: {place} → {departure}")
                    break
        
        # "로/으로/까지" 앞 = 목적지
        to_match = re.search(r"([가-힣]+)(로|으로|까지)", user_message)
        if to_match:
            place = to_match.group(1)
            for key, value in campus_map.items():
                if key in place:
                    destination = value
                    print(f"[셔틀버스] 목적지 발견: {place} → {destination}")
                    break
        
        # 노선 매칭
        routes = []
        if departure and destination:
            # 출발지→목적지 형태로 매칭
            for r in shuttle_data["shuttle_schedules"]:
                route_name = r["route"]
                # "천안→공주" 형태 파싱
                if "→" in route_name:
                    parts = route_name.split("→")
                    route_from = parts[0]
                    route_to = parts[1]
                    
                    if departure in route_from and destination in route_to:
                        routes.append(r)
                        print(f"[셔틀버스] 노선 매칭 성공: {route_name}")
                        break
        
        # 매칭 실패 시 키워드로 폴백
        if not routes:
            print(f"[셔틀버스] 출발/목적지 매칭 실패, 키워드 검색으로 폴백")
            locations = re.findall(r"[가-힣]+", user_message)
            routes = [r for r in shuttle_data["shuttle_schedules"]
                      if any(loc in r["route"] for loc in locations)]
        
        if not routes:
            routes = [shuttle_data["shuttle_schedules"][0]]
        
        print(f"[셔틀버스] 최종 매칭된 노선: {[r['route'] for r in routes]}")

        context = {"service_period": shuttle_data["service_period"], "routes": routes}
        shuttle_ctx = json.dumps(context, ensure_ascii=False)

        prompt = f"""다음은 공주대학교 셔틀버스 운영 정보입니다:
{shuttle_ctx}

위 정보를 참고해 아래 질문에 답해주세요.

[질문]
{user_message}

[답변]
"""
        answer = generate_answer_qwen(prompt, llm_tokenizer, llm_model)
        add_to_history(session_id, "assistant", answer)
        
        client_ip = request.client.host
        save_log(user_input=user_message, matched_paragraphs=[], answer=answer,
                 tokenizer=tokenizer, model=model, client_ip=client_ip)
        return {"response": answer}

    # ========== 7. 공주대 소개 ==========
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
        add_to_history(session_id, "assistant", intro)
        return {"response": intro}

    # ========== 8. 식단 정보 (확장된 메시지 사용) ==========
    if any(k in user_message for k in ("메뉴", "식단", "식당", "점심", "저녁", "아침")):
        campus = "cheonan"
        mapping = {
            "천안": "cheonan", "드림": "dream", "예산": "yesan",
            "은행사": "ehh", "홍익사": "ehh", "해오름": "ehh",
            "비전": "VB", "블룸": "VB",
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
                answer = f"실시간 식단 정보 조회 중 오류 발생: {e}"
                add_to_history(session_id, "assistant", answer)
                return {"response": answer}

        today = datetime.now().strftime("%m월 %d일")
        today_meal = next((m for m in crawl_data.get("meal", [])
                          if today in m.get("date","").strip()), None)

        if not today_meal:
            answer = f"{campus.capitalize()} 캠퍼스 식단 정보가 준비되지 않았습니다."
            add_to_history(session_id, "assistant", answer)
            return {"response": answer}

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

        add_to_history(session_id, "assistant", answer)
        save_log(user_input=user_message, matched_paragraphs=[], answer=answer,
                 tokenizer=tokenizer, model=model, client_ip=request.client.host)
        return {"response": answer}

    # ========== ✅ 9. RAG 검색 (이미 확장된 메시지 사용) ==========
    # user_message는 이미 line 254에서 expanded_message로 교체됨!
    
    q_emb = embed_texts([user_message], tokenizer, model)[0]

    # Milvus 또는 FAISS 검색
    if USE_MILVUS:
        try:
            matched = milvus_client.search(q_emb, top_k=3)
            print(f"[Milvus] 검색 완료: {len(matched)}개 결과")
        except Exception as e:
            print(f"[Milvus] 검색 실패: {e}")
            answer = CANNED_RESPONSE
            add_to_history(session_id, "assistant", answer)
            return {"response": answer}
    else:
        # 차원 검증
        if q_emb.shape[0] != index.d:
            print(f"[오류] 임베딩 차원 불일치! Expected: {index.d}, Got: {q_emb.shape[0]}")
            answer = CANNED_RESPONSE
            add_to_history(session_id, "assistant", answer)
            return {"response": answer}
        
        _, idxs = index.search(q_emb.reshape(1, -1), 3)
        matched = [data[i] for i in idxs[0] if i < len(data)]

    # 키워드 필터링
    keywords = extract_keywords(user_message)
    print(f"[키워드] 추출됨: {keywords}")

    if USE_MILVUS:
        if keywords:
            matched = [
                p for p in matched
                if any(kw.replace(" ", "") in p.get("content", "").replace(" ", "")
                    for kw in keywords)
            ]
            print(f"[Milvus] 키워드 필터링 후: {len(matched)}개")
    else:
        if keywords:
            matched = [
                p for p in matched
                if any(kw.replace(" ", "") in p.get("content", "").replace(" ", "")
                    for kw in keywords)
            ]
            print(f"[FAISS] 키워드 필터링 후: {len(matched)}개")

    # 유사도 확인
    if USE_MILVUS:
        if matched:
            max_sim = max([p.get("score", 0) for p in matched])
            print(f"[Milvus] 최고 유사도: {max_sim:.3f}")
        else:
            max_sim = 0.0
            print(f"[Milvus] 매칭된 문단 없음")
    else:
        matched_with_sim = []
        for p in matched:
            p_emb = embed_texts([p.get("content", "")], tokenizer, model)[0]
            sim = np.dot(q_emb, p_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(p_emb) + 1e-9)
            matched_with_sim.append((p, sim))
        
        matched = [p for p, sim in matched_with_sim if sim >= 0.4]
        sims = [sim for p, sim in matched_with_sim if sim >= 0.4]
        max_sim = max(sims) if sims else 0.0
        
        if matched:
            print(f"[FAISS] 최고 유사도: {max_sim:.3f}, 매칭 문단 수: {len(matched)}")
        else:
            print(f"[FAISS] 매칭 실패 (최고 유사도: {max_sim:.3f})")

    # 매칭 실패 시 기본 응답
    if not matched or max_sim < 0.4:
        print(f"[RAG] 매칭 실패 → 기본 응답 반환")
        answer = CANNED_RESPONSE
        add_to_history(session_id, "assistant", answer)
        return {"response": answer}

    # 프롬프트 생성 및 답변
    prompt = build_prompt(user_message, [matched[0]], conversation_history=history)
    answer = generate_answer_qwen(prompt, llm_tokenizer, llm_model)

    print(f"[답변 생성] 완료 (길이: {len(answer)}자)")

    add_to_history(session_id, "assistant", answer)

    # 로그 저장
    client_ip = get_client_ip(request)
    save_log(
        user_input=user_message,
        matched_paragraphs=[matched[0]], 
        answer=answer,
        tokenizer=tokenizer, 
        model=model, 
        client_ip=client_ip
    )

    return {"response": answer}


@app.post("/api/chat/reset")
async def reset_conversation(request: Request):
    """대화 히스토리 초기화"""
    session_id = get_session_id(request)
    
    if session_id in conversation_history:
        del conversation_history[session_id]
        del session_last_activity[session_id]
    
    return {"message": "대화 기록이 초기화되었습니다."}


@app.get("/api/chat/history")
async def get_conversation_history(request: Request):
    """현재 세션의 대화 히스토리 조회"""
    session_id = get_session_id(request)
    history = conversation_history.get(session_id, [])
    
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"].isoformat()
            }
            for msg in history
        ]
    }