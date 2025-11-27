import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import LLM_MODEL_NAME, CURRENT_CONFIG, ENVIRONMENT

# 환경 정보 출력
print(f"[환경] {ENVIRONMENT} 모드로 실행 중")
print(f"[모델] {LLM_MODEL_NAME} 사용")


def fix_encoding_errors(text: str) -> str:
    """LLM 출력에서 인코딩 오류 교정"""
    error_patterns = {
        "캠�ßer": "캠퍼스",
        "캠ßer": "캠퍼스",
        "캠�": "캠퍼스",
        "�ßer": "퍼스",
        "ßer": "퍼스",
        "대학�": "대학교",
        "학�": "학교",
    }
    
    for wrong, correct in error_patterns.items():
        if wrong in text:
            print(f"[인코딩 교정] '{wrong}' → '{correct}'")
            text = text.replace(wrong, correct)
    
    return text

def load_llm(model_name=LLM_MODEL_NAME):
    """
    Qwen 2.5 모델 로드 (4-bit 양자화)
    config.py에서 환경별 모델 자동 선택
    """
    print(f"[LLM] {model_name} 모델 로딩 중...")
    
    # 4-bit 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print(f"[LLM] {model_name} 모델 로드 완료!")
    return tokenizer, model


def has_foreign_language(text):
    """
    텍스트에 외국어(중국어, 영어, 일본어 등)가 포함되어 있는지 확인
    
    Returns:
        True if foreign language detected, False otherwise
    """
    # 중국어 문자 범위 (간체/번체)
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]'
    
    # 일본어 문자 범위 (히라가나, 가타카나)
    japanese_pattern = r'[\u3040-\u309f\u30a0-\u30ff]'
    
    # 영어 문장 패턴 (3개 이상의 연속된 영어 단어)
    english_sentence_pattern = r'\b[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\b'
    
    if re.search(chinese_pattern, text):
        return True
    if re.search(japanese_pattern, text):
        return True
    if re.search(english_sentence_pattern, text):
        return True
    
    return False


def generate_answer_qwen(prompt, tokenizer, model):
    """
    Qwen 모델로 답변 생성 (✅ 다층 방어: 프롬프트 + 파라미터 + 필터링)
    
    Args:
        prompt: RAG 프롬프트 (chat_utils.py에서 생성)
        tokenizer: Qwen 토크나이저
        model: Qwen 모델
    
    Returns:
        생성된 답변 텍스트 (한국어만)
    """
    # ✅ 1단계 방어: Temperature 낮추기 (더 결정론적으로)
    max_new_tokens = CURRENT_CONFIG['max_new_tokens']
    temperature = 0.3  # ⚠️ 0.7 → 0.3 (외국어 생성 억제)
    top_p = 0.9
    repetition_penalty = CURRENT_CONFIG['repetition_penalty']
    
    try:
        # ✅ 2단계 방어: 매우 강력한 시스템 프롬프트
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant for Kongju National University. You MUST respond ONLY in Korean.

[CRITICAL RULES - STRICTLY ENFORCE]
1. Output language: Korean ONLY (한국어만 사용)
2. BANNED languages: Chinese (中文), English, Japanese (日本語)
3. If you generate ANY non-Korean text, STOP and restart in Korean
4. User questions in foreign languages → Answer in Korean
5. Every single word must be in Korean (Hangul)

답변은 반드시 한국어로만 작성하세요. 중국어, 영어, 일본어 등 외국어는 절대 사용 금지입니다.

Answer concisely in 1-2 sentences in KOREAN."""
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Chat template 적용
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 토크나이징
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # ✅ 3단계 방어: 답변 생성 (낮은 temperature로)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,  # 0.3으로 낮춤
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )
        
        # 입력 부분 제거하고 답변만 디코딩
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        
        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 후처리
        response = response.strip()

        response = fix_encoding_errors(response)
        
        # ✅ 4단계 방어: 외국어 감지 시 재시도 또는 필터링
        if has_foreign_language(response):
            print(f"[WARNING] 외국어 감지됨, 재생성 시도")
            
            # 더 강력한 프롬프트로 재시도
            messages[0]["content"] = """당신은 공주대학교 AI입니다. 

【경고】외국어 사용 절대 금지
- 한국어만 사용
- 중국어 금지
- 영어 금지  
- 일본어 금지

한국어로만 1-2문장으로 답변:"""
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,  # 더 낮게
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty,
                )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            # 여전히 외국어가 있으면 제거
            if has_foreign_language(response):
                print(f"[WARNING] 재생성에도 외국어 포함, 필터링 적용")
                # 한글과 기본 문장부호만 남기기
                response = re.sub(r'[^\uac00-\ud7a3\s.,!?~\-\d]', '', response)
                response = response.strip()
        
        # 응답이 비어있거나 너무 짧으면 기본 메시지
        if not response or len(response) < 3:
            return "해당 정보를 찾을 수 없습니다."
        
        return response
        
    except Exception as e:
        print(f"[ERROR] LLM 답변 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return "답변 생성 중 오류가 발생했습니다."


# 하위 호환성을 위한 별칭
def generate_answer_ollama(prompt, tokenizer=None, model=None):
    """
    기존 Ollama 함수명 호환성 유지
    """
    if tokenizer is None or model is None:
        return "오류: LLM 모델이 초기화되지 않았습니다"
    return generate_answer_qwen(prompt, tokenizer, model)