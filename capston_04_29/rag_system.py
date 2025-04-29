import os
import json
from text_utils import extract_keywords
from embedding_utils import load_embed_model
from search_utils import build_faiss_index, search_similar_paragraph
from llm_utils import load_llm, generate_answer_ollama
from data_loader import load_paragraphs_and_embeddings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    with open("kongju_data.json", "r", encoding="utf-8") as f:
        kongju_data = json.load(f)

    all_text = " ".join([entry['content'] for entry in kongju_data])
    keywords = extract_keywords(all_text)
    print(f"[3] 키워드 추출 완료! 총 {len(keywords)}개")

    paragraphs, paragraph_embeddings = load_paragraphs_and_embeddings()
    print(f"[4] 문단 로드 완료! 총 {len(paragraphs)}개")

    index = build_faiss_index(paragraph_embeddings)
    print("[5] FAISS 인덱스 구축 완료!")

    llm_model, llm_tokenizer = load_llm()
    print("[6] LLM 로드 완료!")

    embed_tokenizer, embed_model = load_embed_model()

    while True:
        user_input = input("\n[질문 입력] (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            break

        matched_paragraph = search_similar_paragraph(
            user_input, index, paragraphs, embed_tokenizer, embed_model, keywords
        )

        prompt = f"""당신은 공주대학교에 관한 질문에만 답변하는 전문 AI입니다.
절대로 다른 대학교(예: 국민대학교, 서울대 등)를 언급하거나 생성하지 마세요.
아래의 질문과 관련 문단을 참고하여 공주대학교 기준으로 정확하게 답변하세요.

[질문]
{user_input}

[관련 문단]
{matched_paragraph}

[답변]
"""

        print(f"\n[검색된 문단]: {matched_paragraph}\n")

        answer = generate_answer_ollama(prompt)
        print(f"[AI 답변]: {answer}")
