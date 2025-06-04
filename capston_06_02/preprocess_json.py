import json
import re
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")

# 제거할 UI 텍스트 목록
IGNORE_KEYWORDS = [
    "본문 바로가기", "주메뉴 바로가기", "슬라이드 정지", "닫기", "LOGIN",
    "이전 슬라이드", "다음 슬라이드", "즐겨찾기", "사이트맵", "TOP", "정지", "재생",
    "오늘 하루 동안 열지 않기", "페이스북 공유하기", "트위터 공유하기", "주소 공유하기",
    "e-총장실", "증명서발급", "개인정보처리방침", "KONGJU NATIONAL UNIVERSITY",
    "Value Creator KNU", "인터넷증명발급", "LMS(원격수업)", "PORTAL", "검색"
]

def clean_text(text: str) -> str:
    for keyword in IGNORE_KEYWORDS:
        text = text.replace(keyword, "")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def split_and_filter_sentences(text: str, min_length: int = 10):
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_length]

def preprocess_and_save(input_path="merged_results.json", output_path="split_results_two.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)["results"]

    processed = []
    sentence_set = set()

    for entry in raw_data:
        raw_text = entry.get("content", "")
        cleaned = clean_text(raw_text)
        split_sentences = split_and_filter_sentences(cleaned)

        title = entry.get("title", "").strip()
        category = entry.get("collection", "").strip()

        for sentence in split_sentences:
            if sentence not in sentence_set:
                processed.append({
                    "title": title,
                    "category": category,
                    "content": sentence
                })
                sentence_set.add(sentence)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": processed}, f, ensure_ascii=False, indent=2)

    print(f"✅ 전처리 완료: {len(processed)}개 문장이 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    preprocess_and_save()
