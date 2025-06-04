import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging
import httpx

from sentence_transformers import SentenceTransformer

# NLTK punkt tokenizer 다운로드
nltk.download("punkt")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1) 텍스트 전처리 유틸리티 (process_json.py 내용)
# --------------------------------------------------------------------------
IGNORE_KEYWORDS = [
    "본문 바로가기", "주메뉴 바로가기", "슬라이드 정지", "닫기", "LOGIN",
    "이전 슬라이드", "다음 슬라이드", "즐겨찾기", "사이트맵", "TOP", "정지", "재생",
    "오늘 하루 동안 열지 않기", "페이스북 공유하기", "트위터 공유하기", "주소 공유하기",
    "e-총장실", "증명서발급", "개인정보처리방침", "KONGJU NATIONAL UNIVERSITY",
    "Value Creator KNU", "인터넷증명발급", "LMS(원격수업)", "PORTAL", "검색"
]

def clean_text(text: str) -> str:
    """
    주어진 텍스트에서 IGNORE_KEYWORDS에 속한 UI 텍스트를 제거하고
    불필요한 공백/줄바꿈을 정리하여 반환합니다.
    """
    for keyword in IGNORE_KEYWORDS:
        text = text.replace(keyword, "")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def split_and_filter_sentences(text: str, min_length: int = 10) -> List[str]:
    """
    주어진 텍스트를 문장 단위로 분리한 뒤, 길이가 min_length 이상인 문장만 반환합니다.
    """
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_length]

def preprocess_and_save(input_path="merged_results.json", output_path="split_results_two.json"):
    """
    merged_results.json 파일을 읽어 각 문단을 clean_text로 전처리하고
    split_and_filter_sentences로 문장 단위로 분리한 뒤, 중복을 제거하여
    split_results_two.json으로 저장합니다.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)["results"]

    processed = []
    sentence_set = set()

    for entry in raw_data:
        raw_text = entry.get("content", "")
        cleaned = clean_text(raw_text)
        split_sentences = split_and_filter_sentences(cleaned)

        title = entry.get("title", "").strip()
        category = entry.get("collection", "").strip()

        for sentence in split_sentences:
            if sentence not in sentence_set:
                processed.append({
                    "title": title,
                    "category": category,
                    "content": sentence
                })
                sentence_set.add(sentence)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": processed}, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 전처리 완료: {len(processed)}개 문장이 '{output_path}'에 저장되었습니다.")

# --------------------------------------------------------------------------
# 2) 임베딩 관련 함수 (embedding_utils.py 내용)
# --------------------------------------------------------------------------
def load_embed_model():
    """
    Sentence-Transformers 기반 임베딩 모델을 로드하여 반환합니다.
    tokenizer는 사용하지 않으므로 None을 반환합니다.
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    return None, model

def embed_texts(texts: List[str], tokenizer, model) -> np.ndarray:
    """
    texts 목록을 모델로 임베딩하여 numpy 배열로 반환합니다.
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

# --------------------------------------------------------------------------
# 3) LLM 호출 관련 함수 (llm_utils.py 내용)
# --------------------------------------------------------------------------
LLM_MODEL_NAME = "mistral:latest"

def generate_answer_ollama(prompt: str) -> str:
    """
    로컬에 실행 중인 Ollama 서버에 POST 요청을 보내 응답을 생성하여 반환합니다.
    """
    friendly_prompt = f"""
당신은 공주대학교 정보를 제공하는 AI 챗봇 포티입니다.
말투는 친절하고 따뜻하게, 이모지를 적절히 사용하여 응답해주세요.

[질문]
{prompt}

[답변]
"""
    try:
        response = httpx.post(
            "http://localhost:11434/v1/generate",
            json={
                "model": LLM_MODEL_NAME,
                "prompt": friendly_prompt,
                "stream": False
            },
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        # “choices → message → content” 구조
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        elif "output" in data:
            return data["output"]
        else:
            return "[오류] Ollama 응답 형식이 예상과 다릅니다."
    except Exception as e:
        logger.error(f"Ollama API 호출 중 오류 발생: {e}")
        return "[오류] Ollama 응답을 가져오지 못했습니다."

# --------------------------------------------------------------------------
# 4) FAISS 인덱스 준비 유틸리티 (data_utils.py 내용)
# --------------------------------------------------------------------------
def load_paragraphs(json_path: str) -> List[dict]:
    """
    split_results_two.json 파일로부터 전처리된 문장(paragraph) 리스트를 읽어 반환합니다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]

def save_faiss_index(index, path: str):
    """
    FAISS 인덱스를 지정된 경로에 저장합니다.
    """
    faiss.write_index(index, path)

def load_faiss_index(path: str):
    """
    지정된 경로에서 FAISS 인덱스를 읽어 반환합니다.
    """
    return faiss.read_index(path)

def prepare_faiss(paragraphs: List[dict],
                  embed_path: str,
                  index_path: str,
                  tokenizer,
                  model) -> faiss.Index:
    """
    paragraphs(문장 목록)에 대해 embedding이 저장된 파일(embed_path)와 인덱스 파일(index_path)이
    존재하면 불러오고, 없으면 embed_texts를 통해 새로 계산하여 저장한 뒤 FAISS Index를 만들어 반환합니다.
    """
    if os.path.exists(embed_path) and os.path.exists(index_path):
        embeddings = np.load(embed_path)
        index = load_faiss_index(index_path)
    else:
        texts = [p["content"] for p in paragraphs]
        embeddings = embed_texts(texts, tokenizer, model)
        np.save(embed_path, embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        save_faiss_index(index, index_path)
    return index

# --------------------------------------------------------------------------
# 5) 챗봇 대화 유틸리티 (chat_utils.py 내용)
# --------------------------------------------------------------------------
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

def is_small_talk(user_input: str) -> str:
    """
    사용자의 입력이 사전에 정의된 인삿말 키워드를 포함하면 해당 응답을 반환하고,
    아니면 None을 반환합니다.
    """
    for key in PREDEFINED_RESPONSES:
        if key in user_input.lower():
            return PREDEFINED_RESPONSES[key]
    return None

def search_similar_paragraphs(query: str,
                              paragraphs: List[dict],
                              tokenizer,
                              model,
                              index,
                              top_k: int = 3) -> List[dict]:
    """
    사용자의 query를 embed_texts로 임베딩하여 FAISS index에서 유사도 검색을 수행하고,
    상위 top_k개의 문단을 반환합니다.
    """
    query_embedding = embed_texts([query], tokenizer, model)[0].reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    return [paragraphs[i] for i in indices[0]]

def build_prompt(user_input: str, matched_paragraphs: List[dict]) -> str:
    """
    검색된 matched_paragraphs를 기반으로 ChatGPT LLM에 입력할 프롬프트를 생성합니다.
    """
    combined_context = "\n".join([p["content"] for p in matched_paragraphs])
    return f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.

- 절대로 다른 대학교(예: 국민대학교, 서울대 등)를 언급하거나 생성하지 마세요.  
- 공주대학교는 캠퍼스별로 위치가 분리되어 있으므로, 실제 행정구역을 정확하게 사용하세요.  
- 특히 "공주시 천안동", "공주광역시 천안동" 같은 잘못된 지명은 사용하지 마세요.  
- 천안캠퍼스는 충청남도 천안시, 예산캠퍼스는 충청남도 예산군, 본교는 충청남도 공주시에 위치합니다.

[질문]
{user_input}

[관련 문단]
{combined_context}

[답변]
"""

def save_log(user_input: str, answer: str, log_dir: str = "logs"):
    """
    사용자 질문과 AI 응답을 파일로 저장합니다.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"log_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"[질문]\n{user_input}\n\n[답변]\n{answer}")

# --------------------------------------------------------------------------
# FastAPI 서버 엔드포인트 정의 (ai_server2.py 내용)
# --------------------------------------------------------------------------
app = FastAPI(title="공주대학교 AI 서버", version="1.0.0")

# 프로젝트 루트 및 데이터 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MERGED_JSON_PATH = os.path.join(PROJECT_ROOT, "merged_results.json")
DATA_PATH = os.path.join(PROJECT_ROOT, "split_results_two.json")
EMBED_PATH = os.path.join(PROJECT_ROOT, "embeddings.npy")
INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss.index")

# 1) split_results_two.json 파일이 없으면 자동 전처리 수행
if not os.path.exists(DATA_PATH):
    if os.path.exists(MERGED_JSON_PATH):
        logger.info(f"'{DATA_PATH}' 파일이 없어 전처리를 시작합니다...")
        preprocess_and_save(input_path=MERGED_JSON_PATH, output_path=DATA_PATH)
    else:
        logger.error(f"전처리에 필요한 '{MERGED_JSON_PATH}' 파일을 찾을 수 없습니다.")
        raise FileNotFoundError(f"'{MERGED_JSON_PATH}' 파일이 필요합니다.")

# 데이터 및 임베딩 모델 로드
data = load_paragraphs(DATA_PATH)
tokenizer, model = load_embed_model()
index = prepare_faiss(data, EMBED_PATH, INDEX_PATH, tokenizer, model)

# 요청 모델 정의
class ChatMessage(BaseModel):
    role: str  # "user" 또는 "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    /api/chat 엔드포인트:
    - 최근에 보내진 user 역할 메시지를 찾아,
    - 사소한 인삿말인지 확인(is_small_talk),
    - 아니라면 FAISS 유사 문단 검색 및 LLM 응답 생성 후 반환.
    """
    # 가장 최근 user 메시지 추출
    user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message provided")

    # 사소한 대화 판별
    response = is_small_talk(user_message)
    if response:
        return {"response": response}

    # 유사 문단 검색
    matched_paragraphs = search_similar_paragraphs(user_message, data, tokenizer, model, index, top_k=3)
    prompt = build_prompt(user_message, matched_paragraphs)

    # Ollama LLM 응답 생성
    answer = generate_answer_ollama(prompt)

    # 로그 저장
    save_log(user_message, answer)

    return {"response": answer}

# --------------------------------------------------------------------------
# 옵션: 직접 실행 시 전처리 도구 호출
# --------------------------------------------------------------------------
if __name__ == "__main__":
    """
    이 스크립트를 직접 실행하면 merged_results.json → split_results_two.json 전처리를 수행합니다.
    단, uvicorn으로 구동할 때도 위에서 자동 전처리가 이미 이루어지므로,
    별도 실행 없이 바로 uvicorn을 실행해도 전처리→임베딩→서버 구동이 모두 자동으로 완료됩니다.
    """
    preprocess_and_save(input_path=MERGED_JSON_PATH, output_path=DATA_PATH)
