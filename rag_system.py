import os
import requests
from bs4 import BeautifulSoup
import re
import nltk
import httpx
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import faiss
from konlpy.tag import Okt

# OpenMP 오류 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# nltk 초기화
nltk.download('punkt')

# 설정값
URL = 'https://ko.wikipedia.org/wiki/서울특별시'
EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL_NAME = 'google/flan-t5-large'

# Okt 형태소 분석기 초기화
okt = Okt()

# --- 크롤링 함수 ---
def crawl_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    for unwanted in soup(['script', 'style', 'header', 'footer', 'nav']):
        unwanted.decompose()
    text = soup.get_text(separator='\n')
    text = re.sub(r'\n+', '\n', text)
    return text

# --- 키워드 추출 ---
def extract_keywords(text, top_n=30):
    words = okt.nouns(text)  # 명사만 추출
    freq_dist = nltk.FreqDist(words)
    keywords = [word for word, freq in freq_dist.most_common(top_n)]
    return keywords

# --- 텍스트를 문단 단위로 분리 ---
def split_into_paragraphs(text):
    paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]
    return paragraphs

# --- 임베딩 모델 로드 ---
def load_embed_model(model_name=EMBED_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# --- 문단 임베딩 ---
def embed_texts(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# --- FAISS 인덱스 구축 ---
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# --- LLM 로드 ---
def load_llm(model_name=LLM_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# --- 질문 받고 가장 비슷한 문단 검색 ---
def search_similar_paragraph(question, index, paragraphs, embed_tokenizer, embed_model, keywords):
    encoded_input = embed_tokenizer(question, return_tensors='pt', truncation=True)
    with torch.no_grad():
        model_output = embed_model(**encoded_input)
    question_embedding = model_output.last_hidden_state.mean(dim=1).numpy()
    D, I = index.search(question_embedding, k=3)

    best_score = -1
    best_paragraph = "(관련 문단 없음)"

    question_words = set(nltk.word_tokenize(question.lower()))
    for idx in I[0]:
        if idx >= len(paragraphs):
            continue
        para = paragraphs[idx]
        para_words = set(nltk.word_tokenize(para.lower()))
        match_score = len(question_words & para_words & set(keywords))
        if match_score > best_score:
            best_score = match_score
            best_paragraph = para

    return best_paragraph

# --- 검색된 문단을 이용해 답변 생성 ---
def generate_answer_ollama(prompt):
    try:
        response = httpx.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False
            },
            timeout=60  # 타임아웃을 60초로 설정
        )
        return response.json()['response']
    except httpx.TimeoutException:
        return "타임아웃 오류가 발생했습니다. 서버가 응답하지 않았습니다."

# --- 메인 실행 흐름 ---
if __name__ == "__main__":
    text = crawl_page(URL)
    print("[1] 크롤링 완료!")

    keywords = extract_keywords(text)
    print(f"[2] 키워드 추출 완료! 총 {len(keywords)}개")
    print("추출 키워드:", keywords)

    paragraphs = split_into_paragraphs(text)
    print(f"[3] 문단 분리 완료! 총 {len(paragraphs)}개 문단")

    embed_tokenizer, embed_model = load_embed_model()
    paragraph_embeddings = embed_texts(paragraphs, embed_tokenizer, embed_model)
    print("[4] 문단 임베딩 완료!")

    index = build_faiss_index(paragraph_embeddings)
    print("[5] FAISS 인덱스 구축 완료!")

    llm_model, llm_tokenizer = load_llm()
    print("[6] LLM 로드 완료!")

    while True:
        user_input = input("\n[질문 입력] (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            break

        matched_paragraph = search_similar_paragraph(user_input, index, paragraphs, embed_tokenizer, embed_model, keywords)
        print(f"\n[검색된 문단]: {matched_paragraph}\n")

        answer = generate_answer_ollama(matched_paragraph)
        print(f"[AI 답변]: {answer}")
