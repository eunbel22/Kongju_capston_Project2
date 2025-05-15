import os
import json
import nltk
import faiss
import numpy as np
from embedding_utils import load_embed_model, embed_texts
from llm_utils import generate_answer_ollama
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# JSON 로드 함수
def load_paragraphs_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["content"] for item in data["results"]]

# FAISS 인덱스 구축 함수
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 가장 유사한 문단 검색
def search_similar_paragraph(query, paragraphs, tokenizer, model, index, top_k=1):
    query_embedding = embed_texts([query], tokenizer, model)[0].reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    return paragraphs[indices[0][0]]

# 메인 실행부
if __name__ == "__main__":
    # 1. 데이터 로드
    paragraphs = load_paragraphs_from_json("merged_results.json")
    print(f"[1] 문단 로드 완료: {len(paragraphs)}개")

    # 2. 임베딩 모델 로드
    tokenizer, model = load_embed_model()
    print("[2] 임베딩 모델 로드 완료")

    # 3. 문단 임베딩
    paragraph_embeddings = embed_texts(paragraphs, tokenizer, model)
    print("[3] 문단 임베딩 완료")

    # 4. FAISS 인덱스 생성
    index = build_faiss_index(paragraph_embeddings)
    print("[4] FAISS 인덱스 구축 완료")

    # 5. 무한 질문 루프
    while True:
        user_input = input("\n[질문 입력] (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            break

        matched_paragraph = search_similar_paragraph(user_input, paragraphs, tokenizer, model, index)

        prompt = f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.

- 절대로 다른 대학교(예: 국민대학교, 서울대 등)를 언급하거나 생성하지 마세요.  
- 공주대학교는 캠퍼스별로 위치가 분리되어 있으므로, 실제 행정구역을 정확하게 사용하세요.  
- 특히 "공주시 천안동", "공주광역시 천안동" 같은 잘못된 지명은 사용하지 마세요.  
- 천안캠퍼스는 충청남도 천안시, 본교는 충청남도 공주시에 위치합니다.

아래 질문과 관련 문단을 참고하여 공주대학교 기준으로 정확하게 답변하세요.

[질문]
{user_input}

[관련 문단]
{matched_paragraph}

[답변]
"""

        print(f"\n[검색된 문단]:\n{matched_paragraph[:300]}...\n")
        answer = generate_answer_ollama(prompt)
        print(f"[AI 답변]:\n{answer}")
