import json
import re

INPUT_FILE = "kongju_data.json"
OUTPUT_FILE = "kongju_cleaned.json"

# 정제 함수 정의
def clean_text(text):
    # 대표적인 잘못된 행정구역 표현 정제
    text = re.sub(r"공주광역시\s*천안동", "충청남도 천안시", text)
    text = re.sub(r"공주광역시", "충청남도 공주시", text)
    text = re.sub(r"천안동", "천안시 동남구", text)
    text = re.sub(r"예산군 예산읍", "충청남도 예산군 예산읍", text)
    return text

# JSON 데이터 로드
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# content 필드 정제
for entry in data:
    entry["content"] = clean_text(entry["content"])

# 정제된 결과 저장
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"정제 완료: {OUTPUT_FILE} 생성됨")
