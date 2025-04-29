# data_loader.py

import os
import json
import numpy as np
from text_utils import split_into_paragraphs
from embedding_utils import load_embed_model, embed_texts

def load_paragraphs_and_embeddings():
    if os.path.exists("paragraphs.json") and os.path.exists("paragraph_embeddings.npy"):
        with open("paragraphs.json", "r", encoding="utf-8") as f:
            paragraphs = json.load(f)
        paragraph_embeddings = np.load("paragraph_embeddings.npy")
        print("[데이터 로드] 문단과 임베딩 로드 완료!")
    else:
        # 🔄 정제된 JSON 사용
        with open("kongju_cleaned.json", "r", encoding="utf-8") as f:
            kongju_data = json.load(f)

        all_text = " ".join([entry['content'] for entry in kongju_data])
        paragraphs = split_into_paragraphs(all_text)

        embed_tokenizer, embed_model = load_embed_model()
        paragraph_embeddings = embed_texts(paragraphs, embed_tokenizer, embed_model)

        with open("paragraphs.json", "w", encoding="utf-8") as f:
            json.dump(paragraphs, f, ensure_ascii=False, indent=4)
        np.save("paragraph_embeddings.npy", paragraph_embeddings)
        print("[데이터 생성] 문단과 임베딩 저장 완료!")

    return paragraphs, paragraph_embeddings
