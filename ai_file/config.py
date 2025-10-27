# config.py
"""
공주대학교 AI 챗봇 프로젝트 설정 파일
개발-서버 환경 분리 및 모델 설정 관리
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# ──────────────────────────────────────────────────────────
# 파일 경로 설정
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent

DATA_PATH = PROJECT_ROOT / "split_results_two.json"
SHUTTLE_PATH = PROJECT_ROOT / "shuttlebus.json"
EMBED_PATH = PROJECT_ROOT / "embeddings.npy"
INDEX_PATH = PROJECT_ROOT / "faiss.index"
PROFANITY_PATH = PROJECT_ROOT / "profanity.json"
SMALL_TALK_PATH = PROJECT_ROOT / "small_talk.json"

# ──────────────────────────────────────────────────────────
# 크롤링 서버 설정
# ──────────────────────────────────────────────────────────
CRAWL_BASE_URL = "http://127.0.0.1:8001"

# ──────────────────────────────────────────────────────────
# 환경 자동 감지 (가장 먼저!)
# ──────────────────────────────────────────────────────────
def detect_environment():
    """
    서버인지 로컬인지 자동 감지
    
    Returns:
        'server': GPU 있고 서버 환경
        'local': 로컬 개발 환경
    """
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # RTX 3090은 24GB VRAM
        if "3090" in gpu_name or gpu_memory > 20:
            return 'server'
        else:
            return 'local'
    else:
        return 'local'


# 환경 감지 (먼저 정의!)
ENVIRONMENT = detect_environment()
IS_SERVER = (ENVIRONMENT == 'server')

print("\n" + "=" * 60)
print(f"🌍 감지된 환경: {ENVIRONMENT.upper()}")
if IS_SERVER:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎮 GPU: {gpu_name}")
print("=" * 60 + "\n")

# ──────────────────────────────────────────────────────────
# LLM 모델 설정 (ENVIRONMENT 정의 후!)
# ──────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'local': {
        'model_name': 'Qwen/Qwen2.5-3B-Instruct',
        'max_new_tokens': 256,
        'temperature': 0.8,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
    },
    'server': {
        'model_name': 'upstage/SOLAR-10.7B-Instruct-v1.0',
        'max_new_tokens': 2048,
        'temperature': 0.8,  #숫자up 더 창의적
        'top_p': 0.9,
        'repetition_penalty': 1.1, #숫자up 반복 방지
    }
}

# 현재 환경의 모델 설정
CURRENT_CONFIG = MODEL_CONFIG.get(ENVIRONMENT, MODEL_CONFIG['local'])
LLM_MODEL_NAME = CURRENT_CONFIG['model_name']

# ──────────────────────────────────────────────────────────
# 임베딩 모델 설정
# ──────────────────────────────────────────────────────────
EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# ──────────────────────────────────────────────────────────
# AI Safeguard 설정
# ──────────────────────────────────────────────────────────
if IS_SERVER:
    # 🚀 서버 환경: 무조건 AI 모델 사용
    DEV_MODE = False
    USE_AI_SAFEGUARD = True
    SAFEGUARD_DEVICE = "auto"
    print("✅ [서버 모드] AI Safeguard 자동 활성화 (GPU)\n")
else:
    # 💻 로컬 환경: 환경변수로 제어
    DEV_MODE = True
    USE_AI_SAFEGUARD = os.getenv("USE_AI_SAFEGUARD", "false").lower() == "true"
    SAFEGUARD_DEVICE = "cpu"
    
    if USE_AI_SAFEGUARD:
        print("🧪 [로컬 모드] AI Safeguard 테스트 (CPU)\n")
    else:
        print("💡 [로컬 모드] JSON 비속어 리스트 사용\n")

# AI Safeguard 모델 설정
SAFEGUARD_MODEL_NAME = "kakaocorp/kanana-safeguard-8b"

# ──────────────────────────────────────────────────────────
# RAG 설정
# ──────────────────────────────────────────────────────────
RAG_CONFIG = {
    'top_k': 3,
    'similarity_threshold': 0.45,
    'keyword_top_n': 30,
}

# ──────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"

# ──────────────────────────────────────────────────────────
# 비속어 필터 설정
# ──────────────────────────────────────────────────────────
PROFANITY_CONFIG = {
    'enabled': True,
    'log_violations': True,
}