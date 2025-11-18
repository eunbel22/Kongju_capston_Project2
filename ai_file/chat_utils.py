#chat_utils.py
import difflib
import os
from datetime import datetime
import numpy as np
from embedding_utils import embed_texts
from nltk import FreqDist
from konlpy.tag import Okt
import json
import re
import torch

# 동의어 사전 로드
SYNONYMS = {
    "공주대학교": ["공주대"],
    "공주대": ["공주대학교"]
}

def replace_synonyms(text):
    for key, syns in SYNONYMS.items():
        for syn in syns:
            if syn in text:
                text = text.replace(syn, key)
    return text

# 프로젝트 구조에 따라 경로 조정
SMALL_TALK_PATH = os.path.join(os.path.dirname(__file__), "data", "small_talk.json")

# 1) JSON에서 매핑 로드
try:
    with open(SMALL_TALK_PATH, "r", encoding="utf-8") as f:
        SMALL_TALK_RESPONSES = json.load(f)
except Exception as e:
    print(f"[chat_utils] small_talk.json 로드 실패: {e}")
    SMALL_TALK_RESPONSES = {}

# ========== 새로 추가: 일상/감정 대화 감지 ========== 
def is_casual_or_emotional(text: str) -> bool:
    """
    일반 대화나 감정 표현인지 판단
    LLM으로 처리할 수 있는 일상적 질문들
    """
    casual_keywords = [
        # 감정 표현
        "배고", "배불", "먹고싶", "맛있",
        "심심", "재미", "지루",
        "힘들", "피곤", "지쳤", "우울", "슬프", "외로",
        "행복", "좋아", "기쁘", "신나",
        
        # 일반 질문
        "뭐하", "어떻게", "어디", "왜",
        "추천", "해줘", "알려줘",
        
        # 학교 관련이지만 너무 일반적
        "수업", "공부", "시험", "과제",
        "대회", "동아리", "행사", "축제"
    ]
    
    text_lower = text.lower().strip()
    return any(kw in text_lower for kw in casual_keywords)

def is_small_talk(user_input: str) -> str | None:
    """
    1) 기존 키워드 매칭 (핵심 패턴)
    2) 유사도 매칭
    """
    text = replace_synonyms(user_input.strip())

    # 1) 부분 매칭
    for key, resp in SMALL_TALK_RESPONSES.items():
        if key in text:
            return resp

    # 2) 유사도 매칭
    candidates = difflib.get_close_matches(text, SMALL_TALK_RESPONSES.keys(),
                                           n=1, cutoff=0.6)
    if candidates:
        return SMALL_TALK_RESPONSES[candidates[0]]

    return None







def build_prompt(user_input, matched_paragraphs):
    normalized_question = replace_synonyms(user_input)
    combined_context = "\n".join([p.get("content", "") for p in matched_paragraphs])
    policy_text = """
[정책 및 지침 예시]
- 다른 대학교 언급 금지
- 위치 정보 정확히 사용할 것
  * 천안캠퍼스: 충청남도 천안시 서북구 천안대로 1223-24(부대동 275
  * 예산캠퍼스: 충청남도 예산군 예산읍 대학로 54(대회리1)
  * 본교(공주캠퍼스): 충청남도 공주시 공주대학로 56(신관동 182)
- 도서관 운영시간은 24시간이 아니야
"""

    prompt = f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.

[관련 문단]
{combined_context}

{policy_text}

[질문]
{normalized_question} 

[답변]
"""
    return prompt


okt = Okt()

def extract_keywords(text, top_n=30):
    text = replace_synonyms(text)
    """
    형태소 분석 결과에서 명사(Noun)와 숫자(Number)만 추출한 뒤,
    길이 2 이상인 단어를 대상으로 빈도 순으로 상위 top_n개를 반환합니다.
    """
    pos_pairs = okt.pos(text, norm=True, stem=True)

    cands = []
    for word, tag in pos_pairs:
        if tag in ("Noun", "Number") and len(word) >= 2:
            cands.append(word)

    freq_dist = FreqDist(cands)
    return [word for word, _ in freq_dist.most_common(top_n)]


def save_log(
    user_input,
    matched_paragraphs,
    answer,
    tokenizer,
    model,
    client_ip,
    log_dir=None,
):
    """
    RAG 과정과 일반 대화를 통합하여 로그를 기록하는 함수입니다.
    """
    try:
        current_file_dir = os.path.dirname(__file__)
        log_dir = os.path.join(current_file_dir, "logs")
        
        os.makedirs(log_dir, exist_ok=True)
        print(f"[LOG] 로그 디렉토리 생성/확인: {log_dir}")
        
        date_str = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(log_dir, f"chat_log_{date_str}.txt")
        print(f"[LOG] 로그 파일 경로: {filename}")

        normalized_input = replace_synonyms(user_input)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        keywords = extract_keywords(user_input)
        query_embedding = embed_texts([user_input], tokenizer, model)[0].reshape(1, -1)

        log_entry = f"[시간] {timestamp}\n"
        log_entry += f"[클라이언트 IP] {client_ip}\n"
        log_entry += f"[질문] {user_input}\n"
        log_entry += f"[키워드] {', '.join(keywords)}\n\n"

        log_entry += "[검색된 문단]\n"
        for i, para in enumerate(matched_paragraphs):
            para_text = para.get("content", "")
            para_category = para.get("category", "")
            para_embedding = embed_texts([para_text], tokenizer, model)[0].reshape(1, -1)
            dot_product = np.dot(query_embedding, para_embedding.T)
            norm_query = np.linalg.norm(query_embedding)
            norm_para = np.linalg.norm(para_embedding)
            similarity = (
                dot_product / (norm_query * norm_para) if norm_query * norm_para != 0 else 0
            )
            log_entry += (
                f"문단 {i+1} (유사도: {similarity[0][0]:.3f}, "
                f"카테고리: {para_category}):\n{para_text}\n\n"
            )

        log_entry += f"[답변]\n{answer}\n"
        log_entry += "-" * 80 + "\n\n\n"

        with open(filename, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        print(f"[LOG] 로그 저장 완료: {filename}")
        
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"[LOG] 파일 크기: {file_size} bytes")
        else:
            print(f"[ERROR] 로그 파일이 생성되지 않음: {filename}")
        
    except Exception as e:
        print(f"[ERROR] 로그 저장 실패: {e}")
        import traceback
        traceback.print_exc()