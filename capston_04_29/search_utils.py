#search_utils.py
# 검색(FAISS 인덱스 생성, 유사 문단 찾기) 함수

# 벡터화된 질문과 키워드가 가장 많이 겹치는 1문장 뽑아 내기
# 질문 임베딩 → FAISS로 유사 문단 3개 검색 → 키워드 겹침 수 기준으로 가장 관련 문단 1개 선택

import faiss
import nltk
import torch

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

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
