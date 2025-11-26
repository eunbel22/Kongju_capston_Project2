# config.py
"""
공주대학교 AI 챗봇 프로젝트 설정 파일
개발-서버 환경 분리 및 모델 설정 관리
"""

import os
import torch

# ──────────────────────────────────────────────────────────
# 환경 설정
# ──────────────────────────────────────────────────────────
IS_GPU_SERVER = torch.cuda.is_available()

# 기존 DEPLOY_ENV 값 읽기 (로컬에서는 development 사용)
RAW_ENV = os.getenv('DEPLOY_ENV', 'development')

# GPU 서버이면 강제로 production 사용 (7B 모델 자동 로드)
if IS_GPU_SERVER:
    ENVIRONMENT = 'production'
else:
    ENVIRONMENT = RAW_ENV
# ──────────────────────────────────────────────────────────
# LLM 모델 설정
# ──────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'development': {
        'model_name': 'Qwen/Qwen2.5-3B-Instruct',
        'max_new_tokens': 128,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.3,
    },
    'production': {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
        'max_new_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.3,
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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# datas 디렉토리
DATAS_DIR = os.path.join(PROJECT_ROOT, "datas")

# datas/output
DATAS_OUTPUT_DIR = os.path.join(DATAS_DIR, "output")

DATA_FILES = [
    os.path.join(DATAS_OUTPUT_DIR, "merged_all_results.json"),
]

# 기타 JSON 파일
SHUTTLE_PATH = os.path.join(DATAS_DIR, "shuttlebus.json")
PROFANITY_PATH = os.path.join(DATAS_DIR, "profanity.json")
SMALL_TALK_PATH = os.path.join(DATAS_DIR, "small_talk.json")

# models/ 디렉토리의 파일들
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EMBED_PATH = os.path.join(MODELS_DIR, "embeddings.npy")
INDEX_PATH = os.path.join(MODELS_DIR, "faiss.index")

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
# 비속어 필터 설정 (✅ 주석 수정)
# ──────────────────────────────────────────────────────────
# GPU 감지
IS_SERVER = torch.cuda.is_available()

# 환경별 비속어 필터 설정
if IS_SERVER:
    # ✅ 서버: Kanana Safeguard 8B 사용 (GPU, 빠름)
    USE_AI_SAFEGUARD = True
    SAFEGUARD_MODEL_NAME = 'kakaocorp/kanana-safeguard-8b'
    SAFEGUARD_DEVICE = 'cuda'
    print("[비속어 필터] 서버 환경 감지 → Kanana Safeguard 8B (GPU) 사용")
else:
    # ✅ 로컬: JSON 사용 (CPU에서 AI 모델은 너무 느림)
    # .env에서 USE_AI_SAFEGUARD=true로 설정하면 AI 모델 사용 가능 (테스트용)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        USE_AI_SAFEGUARD = os.getenv('USE_AI_SAFEGUARD', 'false').lower() == 'true'
    except:
        USE_AI_SAFEGUARD = False
    
    # AI 모델 사용 시 설정 (기본적으로는 사용 안 함)
    SAFEGUARD_MODEL_NAME = 'kakaocorp/kanana-safeguard-8b'
    SAFEGUARD_DEVICE = 'cpu'
    
    if USE_AI_SAFEGUARD:
        print("[비속어 필터] 로컬 환경 → Kanana Safeguard 8B (CPU) 사용 (느릴 수 있음)")
    else:
        print("[비속어 필터] 로컬 환경 → JSON 목록 사용 (권장)")

# ──────────────────────────────────────────────────────────
# Milvus 설정
# ──────────────────────────────────────────────────────────
MILVUS_CONFIG = {
    'host': os.getenv('MILVUS_HOST', '10.37.21.49'),
    'port': 19530,
    'collection_name': 'porty_kongju_docs',
    'vector_dim': 384,
    'vector_field': 'embedding',
    'metric_type': 'IP',
}

USE_MILVUS = os.getenv('USE_MILVUS', 'false').lower() == 'true'

# ──────────────────────────────────────────────────────────
# 디버그 출력
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"환경: {ENVIRONMENT}")
    print(f"LLM 모델: {LLM_MODEL_NAME}")
    print(f"임베딩 모델: {EMBED_MODEL_NAME}")
    print(f"서버 설정: {SERVER_CONFIG[ENVIRONMENT]}")
    print(f"비속어 필터: {'Kanana Safeguard ' + SAFEGUARD_MODEL_NAME.split('/')[-1] if USE_AI_SAFEGUARD else 'JSON'}")
    print(f"  - 모델: {SAFEGUARD_MODEL_NAME if USE_AI_SAFEGUARD else 'N/A'}")
    print(f"  - 디바이스: {SAFEGUARD_DEVICE if USE_AI_SAFEGUARD else 'N/A'}")
    print("=" * 60)
    print("\n📁 파일 경로:")
    print(f"  - SHUTTLE_PATH: {SHUTTLE_PATH}")
    print(f"  - PROFANITY_PATH: {PROFANITY_PATH}")
    print(f"  - SMALL_TALK_PATH: {SMALL_TALK_PATH}")
    print(f"  - EMBED_PATH: {EMBED_PATH}")
    print(f"  - INDEX_PATH: {INDEX_PATH}")
    print(f"  - LOG_DIR: {LOG_DIR}")
    print("=" * 60)
    print("\n🔍 Milvus 설정:")
    print(f"  - 사용: {USE_MILVUS}")
    if USE_MILVUS:
        print(f"  - 호스트: {MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}")
        print(f"  - 컬렉션: {MILVUS_CONFIG['collection_name']}")
    print("=" * 60)