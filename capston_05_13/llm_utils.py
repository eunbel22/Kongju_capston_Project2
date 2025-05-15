from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import httpx
import json

LLM_MODEL_NAME = 'google/flan-t5-large'


def load_llm(model_name=LLM_MODEL_NAME):
    """
    huggingface 기반 LLM 모델 로드
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def generate_answer_ollama(prompt):
    """
    Ollama 서버에 프롬프트를 전송하여 응답을 생성 (stream 모드 사용)
    """
    try:
        with httpx.stream(
            "POST",
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": True
            },
            timeout=120.0  # 타임아웃을 넉넉하게 설정
        ) as response:
            result = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    result += chunk.get("response", "")
            return result.strip()
    except httpx.TimeoutException:
        return "[❌ LLM 응답 타임아웃 오류 발생]"
    except Exception as e:
        return f"[❌ LLM 요청 실패: {str(e)}]"
