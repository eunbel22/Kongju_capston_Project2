import requests
import json

# API 주소는 /search_all로 변경
api_url = "http://10.2.5.220:8000/search_all"

# ✅ Milvus에 저장된 벡터 차원 (예: 2차원 예시)
query_vector = [0.1, 0.2]  # 실제 임베딩 차원에 맞게 수정 필요

# POST 요청 보내기
response = requests.post(api_url, json={"vector": query_vector})
print(f"Status code: {response.status_code}")
print(f"Raw response: {response.text}")

try:
    data = response.json()

    # ✅ 통합 결과 저장
    with open("merged_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("✅ merged_results.json 저장 완료")

except Exception as e:
    print(f"❌ JSON 파싱 실패: {e}")
