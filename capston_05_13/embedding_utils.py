import torch
from transformers import AutoTokenizer, AutoModel

EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def load_embed_model(model_name=EMBED_MODEL_NAME):
    """
    텍스트 임베딩용 모델과 토크나이저 로드
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def embed_texts(texts, tokenizer, model, batch_size=32):
    """
    텍스트 리스트를 임베딩 벡터로 배치 단위로 변환
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)  # 평균 풀링
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0).cpu().numpy()

