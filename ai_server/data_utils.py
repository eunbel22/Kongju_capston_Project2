# 📁 data_utils.py
import os
import json
import faiss
import numpy as np

def load_paragraphs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def load_faiss_index(path):
    return faiss.read_index(path)

def prepare_faiss(paragraphs, embed_path, index_path, tokenizer, model):
    if os.path.exists(embed_path) and os.path.exists(index_path):
        embeddings = np.load(embed_path)
        index = load_faiss_index(index_path)
    else:
        from embedding_utils import embed_texts
        texts = [p["content"] for p in paragraphs]
        embeddings = embed_texts(texts, tokenizer, model)
        np.save(embed_path, embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        save_faiss_index(index, index_path)
    return index
