# 📁 data_utils.py
import os
import json
import faiss
import numpy as np
from embedding_utils import embed_texts, load_embed_model

def load_paragraphs(json_path):
    """
    JSON 파일에서 'results' 필드를 읽어와서 문단 리스트를 반환합니다.
    예외: 'results' 키가 없으면 빈 리스트를 반환하도록 안전성 처리합니다.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", [])
    except Exception as e:
        # 파일 열기나 파싱에 실패하면 빈 리스트 반환
        print(f"[data_utils.load_paragraphs] 오류 발생: {e}")
        return []

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def load_faiss_index(path):
    return faiss.read_index(path)

def prepare_faiss(paragraphs, json_path, embed_path, index_path, tokenizer, model):
    """
    paragraphs: load_paragraphs(json_path)로 로드된 문단 리스트
    json_path: 원본 JSON 파일 경로 (수정 시점을 확인하기 위함)
    embed_path: 임베딩 벡터를 저장할 .npy 파일 경로
    index_path: FAISS 인덱스를 저장할 파일 경로
    tokenizer, model: embed_texts를 위한 임베딩 모델(tokenizer, model)
    """

    # JSON 파일의 최종 수정 시각(초 단위)
    try:
        json_mtime = os.path.getmtime(json_path)
    except Exception as e:
        print(f"[data_utils.prepare_faiss] JSON 파일 수정 시간 조회 오류: {e}")
        json_mtime = None

    # 임베딩 파일과 인덱스 파일이 모두 존재하고, JSON보다 임베딩 파일이 최신이면 재사용
    if json_mtime is not None and os.path.exists(embed_path) and os.path.exists(index_path):
        try:
            embed_mtime = os.path.getmtime(embed_path)
            # embed.npy가 JSON보다 수정 시각이 이후라면, 기존 임베딩·인덱스 재사용
            if embed_mtime >= json_mtime:
                embeddings = np.load(embed_path)
                index = load_faiss_index(index_path)
                return index
        except Exception as e:
            print(f"[data_utils.prepare_faiss] 임베딩/인덱스 파일 조회 오류: {e}")
            # 오류가 발생해도 아래에서 재생성을 진행

    # 여기까지 왔다는 것은:
    # 1) 임베딩 파일이나 인덱스 파일이 없거나
    # 2) JSON 파일이 더 최근에 수정되어 임베딩을 갱신해야 하는 경우
    print("[data_utils.prepare_faiss] 임베딩을 새로 생성합니다...")

    # paragraphs에서 content만 추출하여 embed_texts 실행
    texts = [p.get("content", "") for p in paragraphs]
    if len(texts) == 0:
        # 문단 데이터가 없다면 빈 FAISS 인덱스 생성
        print("[data_utils.prepare_faiss] 문단 데이터가 없습니다. 빈 인덱스를 생성합니다.")
        index = faiss.IndexFlatL2(1)  # 차원 1짜리 빈 인덱스
        np.save(embed_path, np.zeros((0, 1), dtype=np.float32))
        save_faiss_index(index, index_path)
        return index

    # 실제 임베딩 생성
    embeddings = embed_texts(texts, tokenizer, model)  # shape: (N, D)
    # numpy 형태로 저장
    try:
        np.save(embed_path, embeddings)
    except Exception as e:
        print(f"[data_utils.prepare_faiss] embeddings.npy 저장 오류: {e}")

    # FAISS 인덱스 생성 (L2 거리 기준)
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        save_faiss_index(index, index_path)
    except Exception as e:
        print(f"[data_utils.prepare_faiss] FAISS 인덱스 생성/저장 오류: {e}")
        # 실패 시 빈 인덱스 반환
        index = faiss.IndexFlatL2(embeddings.shape[1] if embeddings.ndim > 1 else 1)
    return index