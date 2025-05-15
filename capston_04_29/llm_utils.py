#llm_utils.py
# llm 로드, ollama 답변 생성 함수



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import httpx

LLM_MODEL_NAME = 'google/flan-t5-large'

#hugging face 기반의 llm 로드
def load_llm(model_name=LLM_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# Ollma 서버에 프롬프트 보내서 mistral 모델로 답변 받는 함수
def generate_answer_ollama(prompt):
    try:
        response = httpx.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False
            },
            timeout=60
        )
        return response.json()['response']
    except httpx.TimeoutException:
        return "타임아웃 오류가 발생했습니다. 서버가 응답하지 않았습니다."
