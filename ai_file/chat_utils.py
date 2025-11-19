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
SMALL_TALK_PATH = os.path.join(os.path.dirname(__file__), "datas", "small_talk.json")

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


# ========== ✅ 개선된 대명사 확장 함수 (조사 처리 추가) ==========
def expand_pronouns_with_history(user_input: str, conversation_history: list) -> str:
    """
    대명사를 이전 대화 맥락으로 확장
    
    예: "거기서 뭐 배워?" + 이전 대화(소프트웨어학과) → "소프트웨어학과에서 뭐 배워?"
    """
    if not conversation_history or len(conversation_history) < 2:
        return user_input
    
    # 대명사 패턴 (확장)
    pronouns = ["거기", "그거", "그곳", "여기", "이거", "저기", "그것", "그때", "그날"]
    
    # 대명사가 없으면 그대로 반환
    if not any(p in user_input for p in pronouns):
        return user_input
    
    # 최근 대화에서 명사 추출 (최근 3턴 = 6개 메시지)
    recent_messages = conversation_history[-6:]
    
    # 주요 키워드 찾기 (우선순위: 복합명사 > 단일명사)
    keywords = []
    for msg in recent_messages:
        content = msg.get("content", "")
        
        # 1️⃣ 캠퍼스명 우선 (가장 구체적)
        for campus in ["공주캠퍼스", "천안캠퍼스", "예산캠퍼스"]:
            if campus in content:
                keywords.append(campus)
        
        # 2️⃣ 복합명사 패턴 (학과, 건물명 등)
        patterns = [
            r'([가-힣]+캠퍼스)',
            r'([가-힣]+학과)',
            r'([가-힣]+전공)',
            #r'([가-힣]+대학)',
            r'([가-힣]+관)',
            r'([가-힣]+도서관)',
            r'([가-힣]+센터)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content)
            keywords.extend(matches)
        
        # 3️⃣ 지역명 백업 (복합명사가 없을 때)
        if not keywords:
            for location in ["공주", "천안", "예산"]:
                if location in content:
                    keywords.append(location)
    
    if not keywords:
        return user_input
    
    # 중복 제거 + 길이순 정렬 (긴 것 = 더 구체적)
    keywords_unique = sorted(set(keywords), key=lambda x: len(x), reverse=True)
    main_keyword = keywords_unique[0]  # 가장 긴(구체적인) 키워드 선택
    
    # 🔧 대명사 + 조사를 키워드로 교체
    expanded = user_input
    for pronoun in pronouns:
        # 대명사 + 조사 패턴 매칭 (서, 는, 가, 을, 를, 에, 도, 의, 와, 과, 로 등)
        pattern = f"{pronoun}[서는가을를에도의와과로부터까지만]?"
        if re.search(pattern, expanded):
            expanded = re.sub(pattern, main_keyword, expanded)
            print(f"[대명사 확장] '{user_input}' → '{expanded}'")
            break
        elif pronoun in expanded:
            expanded = expanded.replace(pronoun, main_keyword)
            print(f"[대명사 확장] '{user_input}' → '{expanded}'")
            break
    
    return expanded

# ========== ✅ 새로 추가: 문단 길이 제한 함수 ==========
def truncate_paragraph(content, max_length=2000):
    """
    문단이 너무 길면 자르기
    
    Args:
        content: 원본 문단 내용
        max_length: 최대 글자 수 (기본 2000자)
    
    Returns:
        잘린 문단 (필요시 "...(생략)" 추가)
    """
    if len(content) <= max_length:
        return content
    
    # 문장 단위로 자르기 시도
    sentences = content.split('.')
    truncated = ""
    
    for sentence in sentences:
        if len(truncated) + len(sentence) + 1 <= max_length:
            truncated += sentence + "."
        else:
            break
    
    # 문장 단위로 안 되면 글자 수로 자르기
    if not truncated:
        truncated = content[:max_length]
    
    return truncated + "...(생략)"


# ========== ✅ 대화 히스토리 통합 프롬프트 ==========
def build_prompt(user_input, matched_paragraphs, conversation_history=None):
    """
    RAG 프롬프트 생성 (✅ 대화 히스토리 통합)
    
    Args:
        user_input: 현재 사용자 질문
        matched_paragraphs: 검색된 문단들
        conversation_history: 최근 대화 히스토리 (optional)
    """
    normalized_question = replace_synonyms(user_input)
    
    # ✅ 첫 번째 문단만 사용
    if not matched_paragraphs:
        context = ""
    else:
        context = matched_paragraphs[0].get("content", "")
        # 너무 길면 자르기 (1500자)
        if len(context) > 1500:
            context = context[:1500] + "..."
    
    # ✅ 대화 히스토리가 있으면 포함
    if conversation_history and len(conversation_history) > 0:
        history_text = "\n이전 대화:\n"
        for msg in conversation_history[-4:]:  # 최근 2턴 (4개 메시지)
            role = "사용자" if msg["role"] == "user" else "포티"
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""정보: {context}

{history_text}

현재 질문: {normalized_question}

위 정보와 이전 대화를 참고하여 질문에 한국어로 1-2문장만 답하세요.
답변:"""
    else:
        # 히스토리 없으면 기존 방식
        prompt = f"""정보: {context}

질문: {normalized_question}

위 정보로 질문에 한국어로 1-2문장만 답하세요.
답변:"""
    
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
            
            # ✅ 로그에도 긴 문단은 잘라서 기록 (500자)
            para_text_for_log = truncate_paragraph(para_text, max_length=500)
            
            # 원본으로 임베딩 생성
            para_embedding = embed_texts([para_text], tokenizer, model)[0].reshape(1, -1)
            dot_product = np.dot(query_embedding, para_embedding.T)
            norm_query = np.linalg.norm(query_embedding)
            norm_para = np.linalg.norm(para_embedding)
            similarity = (
                dot_product / (norm_query * norm_para) if norm_query * norm_para != 0 else 0
            )
            log_entry += (
                f"문단 {i+1} (유사도: {similarity[0][0]:.3f}, "
                f"카테고리: {para_category}):\n{para_text_for_log}\n\n"
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