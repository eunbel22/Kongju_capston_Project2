# 📁 chat_utils.py
import difflib
import os
from datetime import datetime
import numpy as np
from embedding_utils import embed_texts
from nltk import FreqDist
from konlpy.tag import Okt
import json





# 프로젝트 구조에 따라 경로 조정
SMALL_TALK_PATH = os.path.join(os.path.dirname(__file__), "small_talk.json")

# 1) JSON에서 매핑 로드
try:
    with open(SMALL_TALK_PATH, "r", encoding="utf-8") as f:
        SMALL_TALK_RESPONSES = json.load(f)
except Exception as e:
    print(f"[chat_utils] small_talk.json 로드 실패: {e}")
    SMALL_TALK_RESPONSES = {}

def is_small_talk(user_input: str) -> str | None:
    """
    1) 가장 먼저 키워드(키)에 substring 매칭을 시도합니다.
    2) 없으면 difflib.get_close_matches로 유사한 키를 찾아봅니다.
    3) 유사도가 cutoff 이상인 경우 해당 응답을 반환.
    """
    text = user_input.strip()

    # 1) 부분 매칭
    for key, resp in SMALL_TALK_RESPONSES.items():
        if key in text:
            return resp

    # 2) 유사도 매칭
    #    keys 목록 중에서 가장 비슷한 키 하나 찾기
    candidates = difflib.get_close_matches(text, SMALL_TALK_RESPONSES.keys(),
                                           n=1, cutoff=0.6)
    if candidates:
        return SMALL_TALK_RESPONSES[candidates[0]]

    return None

def build_prompt(user_input, matched_paragraphs):
    combined_context = "\n".join([p.get("content", "") for p in matched_paragraphs])
    policy_text = """
[정책 및 지침 예시]
- 다른 대학교 언급 금지
- 위치 정보 정확히 사용할 것
  * 천안캠퍼스: 충청남도 천안시
  * 예산캠퍼스: 충청남도 예산군
  * 본교(공주캠퍼스): 충청남도 공주시
- 도서관 운영시간은 24시간이 아니야
"""

    prompt = f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.

[관련 문단]
{combined_context}

{policy_text}

[질문]
{user_input}

[답변]
"""
    return prompt


okt = Okt()

'''
def extract_keywords(text, top_n=30):
    """
    명사 기반 키워드 추출
    """
    words = okt.nouns(text)
    freq_dist = FreqDist(words)
    return [word for word, freq in freq_dist.most_common(top_n)]
'''


def extract_keywords(text, top_n=30):
    """
    형태소 분석 결과에서 명사(Noun)와 숫자(Number)만 추출한 뒤,
    길이 2 이상인 단어를 대상으로 빈도 순으로 상위 top_n개를 반환합니다.
    """
    # 1) 형태소 분석 → (단어, 품사) 튜플 목록 얻기
    pos_pairs = okt.pos(text, norm=True, stem=True)

    # 2) 명사(Noun) 또는 숫자(Number)만 필터링, 길이 >= 2
    cands = []
    for word, tag in pos_pairs:
        if tag in ("Noun", "Number") and len(word) >= 2:
            cands.append(word)

    # 3) 빈도 분석
    freq_dist = FreqDist(cands)

    # 4) 가장 빈도가 높은 top_n개 단어 리스트로 반환
    return [word for word, _ in freq_dist.most_common(top_n)]


def save_log(
    user_input,
    matched_paragraphs,
    answer,
    tokenizer,
    model,
    client_ip,
    log_dir="PortyProject/ai_server/logs",
):
    """
    RAG 과정과 일반 대화를 통합하여 로그를 기록하는 함수입니다.
    - user_input: 사용자가 보낸 질문 문자열
    - matched_paragraphs: 검색된 문단들의 리스트 (각 문단은 dict 형태로 'content'와 'category' 포함)
    - answer: 최종 생성된 답변 문자열
    - tokenizer, model: 임베딩 생성에 사용할 토크나이저와 모델
    - client_ip: 요청을 보낸 클라이언트의 IP 주소
    - log_dir: 로그 파일을 저장할 디렉토리 (일별 파일로 누적 기록)
    """
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    filename = os.path.join(log_dir, f"chat_log_{date_str}.txt")

    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 질문 키워드 추출
    keywords = extract_keywords(user_input)

    # 질문 임베딩 생성
    query_embedding = embed_texts([user_input], tokenizer, model)[0].reshape(1, -1)

    # 로그 항목 구성
    log_entry = f"[시간] {timestamp}\n"
    log_entry += f"[클라이언트 IP] {client_ip}\n"
    log_entry += f"[질문] {user_input}\n"
    log_entry += f"[키워드] {', '.join(keywords)}\n\n"

    # 검색된 문단 및 유사도 계산
    log_entry += "[검색된 문단]\n"
    for i, para in enumerate(matched_paragraphs):
        para_text = para.get("content", "")
        para_category = para.get("category", "")
        # 문단 임베딩 생성
        para_embedding = embed_texts([para_text], tokenizer, model)[0].reshape(1, -1)
        # 코사인 유사도 계산
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

    # 답변 기록
    log_entry += f"[답변]\n{answer}\n"
    log_entry += "-" * 80 + "\n\n\n"

    # 파일에 append 모드로 작성
    with open(filename, "a", encoding="utf-8") as f:
        f.write(log_entry)


