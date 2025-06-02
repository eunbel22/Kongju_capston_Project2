# 📁 chat_utils.py
import os
from datetime import datetime
from embedding_utils import embed_texts

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
        if key in user_input.lower():
            return PREDEFINED_RESPONSES[key]
    return None

def search_similar_paragraphs(query, paragraphs, tokenizer, model, index, top_k=3):
    query_embedding = embed_texts([query], tokenizer, model)[0].reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    return [paragraphs[i] for i in indices[0]]

def build_prompt(user_input, matched_paragraphs):
    combined_context = "\n".join([p["content"] for p in matched_paragraphs])
    return f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.

- 절대로 다른 대학교(예: 국민대학교, 서울대 등)를 언급하거나 생성하지 마세요.  
- 공주대학교는 캠퍼스별로 위치가 분리되어 있으므로, 실제 행정구역을 정확하게 사용하세요.  
- 특히 \"공주시 천안동\", \"공주광역시 천안동\" 같은 잘못된 지명은 사용하지 마세요.  
- 천안캠퍼스는 충청남도 천안시, 예산캠퍼스는 충청남도 예산군, 본교는 충청남도 공주시에 위치합니다.

[질문]
{user_input}

[관련 문단]
{combined_context}

[답변]
"""

def save_log(user_input, answer, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"log_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"[질문]\n{user_input}\n\n[답변]\n{answer}")
