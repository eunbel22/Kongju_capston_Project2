from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import httpx
import json

# 모델 이름 수정
LLM_MODEL_NAME = 'mistral'


def load_llm(model_name=LLM_MODEL_NAME):
    """
    huggingface 기반 LLM 모델 로드 (미사용 시 주석 가능)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def generate_answer_ollama(prompt):
    """
    Ollama 서버에 프롬프트를 전송하여 응답을 생성 (stream 모드 사용)
    """
    friendly_prompt = f"""
    당신은 공주대학교 정보를 제공하는 AI 챗봇 포티입니다.
    말투는 친절하고 따뜻하게, 이모지를 적절히 사용하여 응답해주세요.

    [질문]
    {prompt}

    [답변]
    """
    try:
        with httpx.stream(
            "POST",
            "http://localhost:11434/api/generate",
            json={
                "model": "EEVE-Korean",
                "prompt": friendly_prompt,
                "stream": True
            },
            timeout=300.0
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