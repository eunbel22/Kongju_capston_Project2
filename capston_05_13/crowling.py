import requests
import json

# 팀원이 제공한 API 주소
api_url = "http://<팀원_IP>:8000/search"

# 검색할 벡터 예시 (384차원)
query_vector = [0.1] * 384  # 실제로는 문장을 임베딩한 벡터로 대체

# POST 요청
response = requests.post(api_url, json={"vector": query_vector})

# 결과 JSON 추출
data = response.json()

# JSON 파일로 저장
with open("search_result.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ search_result.json 저장 완료")
