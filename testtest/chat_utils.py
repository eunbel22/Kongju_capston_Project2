# 📁 chat_utils.py
import os
from datetime import datetime
import numpy as np
from embedding_utils import embed_texts
from nltk import FreqDist
from konlpy.tag import Okt


PREDEFINED_RESPONSES = {
    "안녕": "안녕하세요! 저는 공주대학교 AI 도우미, 포티(Porty)입니다 😊",
    "하이": "하이~ 반가워요! 저는 포티예요. 공주대학교에 대해 궁금한 게 있나요?",
    "안녕하세요": "네, 안녕하세요! 공주대학교에 대해 무엇을 도와드릴까요?",
    "잘 지냈어": "네! 포티는 항상 대기 중이에요 😊 무엇이 궁금하신가요?",
    "이름이 뭐야": "제 이름은 포티(Porty)입니다. 공주대학교에 대해 무엇이든 알려드릴게요!",
    "누구야": "저는 공주대학교 정보를 알려주는 AI 포티예요.",
    "고마워": "별말씀을요! 더 궁금한 게 있으면 언제든지 물어보세요 🙌",
    "감사": "감사합니다! 도움이 되었다니 기쁘네요 :)",
    "수고했어": "감사합니다! 포티는 언제나 도와드릴 준비가 되어 있어요.",
    "잘했어": "칭찬 감사합니다! 더 정확하게 답변할 수 있도록 노력할게요.",
    "바보": "포티는 아직 많이 배우는 중이에요 😅 더 나은 답변을 위해 노력할게요!",
    "심심해": "그럴 땐 공주대학교의 다양한 동아리나 행사 정보를 찾아보는 건 어때요?",
    "재밌는 이야기": "음... 포티는 주로 공주대학교 정보에 집중하고 있지만, 궁금한 게 있다면 도와드릴게요!",
    "무슨 일 해": "저는 공주대학교에 대한 정보와 도움을 드리는 챗봇, 포티예요!",
    "포티": "네! 포티가 여기 있어요 😊 무엇이 궁금하신가요?",
    "도와줘": "물론이죠! 공주대학교에 대해 궁금한 걸 말씀해 주세요.",
    "메뉴 알려줘": "식단표를 원하시는 건가요? 어떤 캠퍼스 식단이 궁금하신가요?",
}


def is_small_talk(user_input):
    for key in PREDEFINED_RESPONSES:
        if key in user_input:
            return PREDEFINED_RESPONSES[key]
    return None

def search_similar_paragraphs(query, paragraphs, tokenizer, model, index, top_k=3):
    query_embedding = embed_texts([query], tokenizer, model)[0].reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    return [paragraphs[i] for i in indices[0] if i < len(paragraphs)]

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


