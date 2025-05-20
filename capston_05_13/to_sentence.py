import json
import re

# 문장 분리 함수
def split_into_sentences(text):
    # 기본적인 마침표 기준으로 분리하되, 공백 포함 정리
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

# 기존 파일 열기
with open("merged_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 결과 저장용 리스트
split_results = []

# 각 문단을 문장으로 분리
for item in data["results"]:
    title = item.get("title", "")
    content = item.get("content", "")
    sentences = split_into_sentences(content)
    for sentence in sentences:
        split_results.append({
            "title": title,
            "content": sentence
        })

# 새 파일로 저장
with open("split_results.json", "w", encoding="utf-8") as f:
    json.dump({"results": split_results}, f, ensure_ascii=False, indent=2)

print(f"✅ 총 {len(split_results)}개 문장으로 분할 완료")
