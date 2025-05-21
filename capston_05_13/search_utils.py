import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_faiss_index(paragraphs):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(paragraphs).toarray().astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, vectorizer

def search_similar_paragraph(query, index, vectorizer, paragraphs, k=5, raw_k=20):
    query_vec = vectorizer.transform([query]).toarray().astype('float32')

    # 후보 문장 많이 확보
    D, I = index.search(query_vec, raw_k)

    results = []
    seen = set()

    for idx in I[0]:
        if idx < len(paragraphs):
            para = paragraphs[idx]
            if para not in seen:
                results.append(para)
                seen.add(para)
        if len(results) == k:
            break

    return results

