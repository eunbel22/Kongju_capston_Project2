import nltk
from nltk.tokenize import sent_tokenize
import json

nltk.download("punkt")

def preprocess_documents(documents):
    preprocessed = []
    for doc in documents:
        if isinstance(doc, str):
            sentences = sent_tokenize(doc.strip())
            preprocessed.append(" ".join(sentences))
    return preprocessed

def load_paragraphs_and_embeddings(json_path="merged_results.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    paragraphs = [item["paragraph"] for item in data]
    embeddings = [item["embedding"] for item in data]
    return paragraphs, embeddings
