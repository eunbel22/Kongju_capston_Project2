import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import LLM_MODEL_NAME, CURRENT_CONFIG, ENVIRONMENT

# 환경 정보 출력
print(f"[환경] {ENVIRONMENT} 모드로 실행 중")
print(f"[모델] {LLM_MODEL_NAME} 사용")

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


def generate_answer_qwen(prompt, tokenizer, model):
    """
    Qwen 모델로 답변 생성 (✅ chat template 사용)
    
    Args:
        prompt: RAG 프롬프트 (chat_utils.py에서 생성)
        tokenizer: Qwen 토크나이저
        model: Qwen 모델
    
    Returns:
        생성된 답변 텍스트
    """
    # config에서 파라미터 가져오기
    max_new_tokens = CURRENT_CONFIG['max_new_tokens']
    temperature = CURRENT_CONFIG['temperature']
    top_p = CURRENT_CONFIG['top_p']
    repetition_penalty = CURRENT_CONFIG['repetition_penalty']
    
    try:
        # ✅ Qwen용 chat template 사용
        messages = [
            {
                "role": "system",
                "content": "당신은 공주대학교 정보를 제공하는 AI입니다. 질문에 한국어로 간단명료하게 1-2문장으로만 답변하세요."
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
        
        # 답변 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
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
        
        # 간단한 후처리
        response = response.strip()
        
        # 응답이 비어있으면 기본 메시지
        if not response or len(response) < 3:
            return "정보를 찾을 수 없습니다."
        
        return response
        
    except Exception as e:
        print(f"[ERROR] LLM 답변 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return "답변 생성 중 오류가 발생했습니다."


# 하위 호환성을 위한 별칭 (기존 코드에서 generate_answer_ollama 호출하는 부분 대응)
def generate_answer_ollama(prompt, tokenizer=None, model=None):
    """
    기존 Ollama 함수명 호환성 유지
    주의: tokenizer와 model을 반드시 전달해야 함
    """
    if tokenizer is None or model is None:
        return "오류: LLM 모델이 초기화되지 않았습니다"
    return generate_answer_qwen(prompt, tokenizer, model)