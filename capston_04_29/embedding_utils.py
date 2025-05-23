# 임베딩 모델 로딩, 텍스트 임베딩 함수

import torch
from transformers import AutoTokenizer, AutoModel

EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def load_embed_model(model_name=EMBED_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def embed_texts(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# 질문을 컴퓨터가 알아들을 수 있도록 벡터화 한것
# 문장을 MiniLM 모델을 통해 384차원 의미 벡터로 변환

