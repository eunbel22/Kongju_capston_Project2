# config.py
"""
공주대학교 AI 챗봇 프로젝트 설정 파일
개발-서버 환경 분리 및 모델 설정 관리
"""

import os

# ──────────────────────────────────────────────────────────
# 환경 설정
# ──────────────────────────────────────────────────────────
ENVIRONMENT = os.getenv('DEPLOY_ENV', 'development')  # 'development' or 'production'

# ──────────────────────────────────────────────────────────
# LLM 모델 설정
# ──────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'development': {
        'model_name': 'Qwen/Qwen2.5-3B-Instruct',
        'max_new_tokens': 128,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
    },
    'production': {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
        'max_new_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
    }
}

# 현재 환경의 모델 설정
CURRENT_CONFIG = MODEL_CONFIG.get(ENVIRONMENT, MODEL_CONFIG['development'])
LLM_MODEL_NAME = CURRENT_CONFIG['model_name']

# ──────────────────────────────────────────────────────────
# 임베딩 모델 설정
# ──────────────────────────────────────────────────────────
EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# ──────────────────────────────────────────────────────────
# 서버 설정
# ──────────────────────────────────────────────────────────
SERVER_CONFIG = {
    'development': {
        'host': '127.0.0.1',
        'port': 8000,
        'reload': True,
    },
    'production': {
        'host': '0.0.0.0',
        'port': 8000,
        'reload': False,
    }
}

# ──────────────────────────────────────────────────────────
# 파일 경로 설정
# ──────────────────────────────────────────────────────────
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_ROOT, "split_results_two.json")
SHUTTLE_PATH = os.path.join(PROJECT_ROOT, "shuttlebus.json")
EMBED_PATH = os.path.join(PROJECT_ROOT, "embeddings.npy")
INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss.index")
PROFANITY_PATH = os.path.join(PROJECT_ROOT, "profanity.json")
SMALL_TALK_PATH = os.path.join(PROJECT_ROOT, "small_talk.json")

# ──────────────────────────────────────────────────────────
# 크롤링 서버 설정
# ──────────────────────────────────────────────────────────
CRAWL_BASE_URL = "http://127.0.0.1:8001"

# ──────────────────────────────────────────────────────────
# RAG 설정
# ──────────────────────────────────────────────────────────
RAG_CONFIG = {
    'top_k': 3,  # FAISS 검색 결과 개수
    'similarity_threshold': 0.45,  # 코사인 유사도 임계값
    'keyword_top_n': 30,  # 키워드 추출 개수
}

# ──────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ──────────────────────────────────────────────────────────
# 디버그 출력
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"환경: {ENVIRONMENT}")
    print(f"LLM 모델: {LLM_MODEL_NAME}")
    print(f"임베딩 모델: {EMBED_MODEL_NAME}")
    print(f"서버 설정: {SERVER_CONFIG[ENVIRONMENT]}")
    print("=" * 60)