# profanity_filter.py
"""
비속어 필터링 시스템
- Local: JSON 기반 (빠름)
- Server: Kanana Safeguard 8B (정확함, Classification 전용)
"""
import torch
import re
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import (
    USE_AI_SAFEGUARD,
    SAFEGUARD_MODEL_NAME,
    SAFEGUARD_DEVICE,
    PROFANITY_PATH
)


# ============================================
# JSON 기반 필터 (Local)
# ============================================

def load_profanity_list():
    """profanity.json에서 비속어 목록 로드"""
    try:
        with open(PROFANITY_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("bad_words", [])
    except Exception as e:
        print(f"[WARNING] profanity.json 로드 실패: {e}")
        return []


def contains_profanity_json(text: str, profanity_list: list) -> bool:
    """JSON 목록 기반 비속어 감지"""
    t = text.lower()
    for bad in profanity_list:
        if re.search(rf"\b{re.escape(bad)}\b", t):
            return True
    return False


# ============================================
# AI 모델 기반 필터 (Server) - 8B Classification
# ============================================

safeguard_model = None
safeguard_tokenizer = None

def load_safeguard_model():
    """Kanana Safeguard 8B 모델 로드 (4-bit 양자화, Classification 전용)"""
    global safeguard_model, safeguard_tokenizer
    
    if not USE_AI_SAFEGUARD:
        return None, None
    
    if safeguard_model is not None:
        return safeguard_tokenizer, safeguard_model
    
    print(f"[비속어 필터] Kanana Safeguard 8B 모델 로딩 중... ({SAFEGUARD_DEVICE})")
    
    try:
        from transformers import BitsAndBytesConfig
        
        # 4-bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        safeguard_tokenizer = AutoTokenizer.from_pretrained(SAFEGUARD_MODEL_NAME)
        
        if SAFEGUARD_DEVICE == 'cuda':
            # ✅ GPU: Classification 모델로 로드 (CausalLM 아님!)
            safeguard_model = AutoModelForSequenceClassification.from_pretrained(
                SAFEGUARD_MODEL_NAME,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            # CPU: 양자화 없이 로드 (느림)
            safeguard_model = AutoModelForSequenceClassification.from_pretrained(
                SAFEGUARD_MODEL_NAME,
                trust_remote_code=True
            )
            safeguard_model = safeguard_model.to(SAFEGUARD_DEVICE)
        
        safeguard_model.eval()  # ✅ 평가 모드로 전환
        print(f"[비속어 필터] Kanana Safeguard 8B 로드 완료! ({SAFEGUARD_DEVICE})")
        return safeguard_tokenizer, safeguard_model
        
    except Exception as e:
        print(f"[ERROR] Safeguard 모델 로드 실패: {e}")
        print("[INFO] JSON 필터로 fallback 됩니다.")
        return None, None


def contains_profanity_ai(text: str, threshold: float = 0.5) -> bool:
    """
    AI 모델 기반 비속어 감지 (Classification)
    
    Args:
        text: 검사할 텍스트
        threshold: 유해 판정 임계값 (0.5 = 50% 확률 이상이면 유해)
    
    Returns:
        True: 유해한 텍스트
        False: 안전한 텍스트
    """
    global safeguard_model, safeguard_tokenizer
    
    if safeguard_model is None or safeguard_tokenizer is None:
        return False
    
    try:
        # ✅ 텍스트 토크나이징 (프롬프트 없이 직접 입력)
        inputs = safeguard_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(SAFEGUARD_DEVICE)
        
        # ✅ 분류 수행 (generate 아님!)
        with torch.no_grad():
            outputs = safeguard_model(**inputs)
            logits = outputs.logits
            
            # 소프트맥스로 확률 변환
            probabilities = torch.softmax(logits, dim=-1)
            
            # ✅ Label 0 = 안전, Label 1 = 유해
            harmful_prob = probabilities[0][1].item()
        
        is_harmful = harmful_prob >= threshold
        
        # 디버그 로그
        if is_harmful:
            print(f"[비속어 감지] '{text[:30]}...' → 유해 확률: {harmful_prob:.2%}")
        
        return is_harmful
        
    except Exception as e:
        print(f"[ERROR] Safeguard 모델 실행 실패: {e}")
        return False


# ============================================
# 통합 인터페이스
# ============================================

class ProfanityFilter:
    """환경별 비속어 필터 통합 클래스"""
    
    def __init__(self):
        self.use_ai = USE_AI_SAFEGUARD
        
        if self.use_ai:
            # AI 모델 로드
            self.tokenizer, self.model = load_safeguard_model()
            if self.model is None:
                print("[WARNING] AI 모델 로드 실패, JSON 필터 사용")
                self.use_ai = False
                self.profanity_list = load_profanity_list()
        else:
            # JSON 목록 로드
            self.profanity_list = load_profanity_list()
            print(f"[비속어 필터] JSON 목록 사용 ({len(self.profanity_list)}개 단어)")
    
    def contains_profanity(self, text: str) -> bool:
        """비속어 포함 여부 확인"""
        if self.use_ai:
            return contains_profanity_ai(text)
        else:
            return contains_profanity_json(text, self.profanity_list)
    
    def __call__(self, text: str) -> bool:
        """편의 메서드"""
        return self.contains_profanity(text)


# ============================================
# 전역 필터 인스턴스 (서버 시작 시 한 번만 로드)
# ============================================
print("[초기화] 비속어 필터 초기화 중...")
profanity_filter = ProfanityFilter()
print("[초기화] 비속어 필터 준비 완료!")


# 편의 함수
def contains_profanity(text: str) -> bool:
    """비속어 포함 여부 확인 (전역 함수)"""
    return profanity_filter.contains_profanity(text)