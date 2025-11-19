#!/usr/bin/env python3
"""
data, sw_data, swknu_data 폴더의 JSON 파일을 각각 병합하여
output 폴더에 개별 JSON으로 저장하는 스크립트
"""

import json
import os
from pathlib import Path

# ---------------------------------------------------------
# JSON 로드 함수
# ---------------------------------------------------------
def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ JSON 로드 실패 - {filepath} : {e}")
        return None

# ---------------------------------------------------------
# 파일명 → 카테고리 추출
# ---------------------------------------------------------
def extract_category_from_filename(filename):
    name = filename.replace('.json', '')
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name

# ---------------------------------------------------------
# 폴더별 JSON 병합 함수
# ---------------------------------------------------------
def merge_folder(folder_path, output_file):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ 폴더 없음: {folder}")
        return

    # 출력 폴더 자동 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 기존 데이터 로드
    merged_results = []
    if os.path.exists(output_file):
        print(f"📂 기존 병합 파일 로드: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            old = json.load(f)
            merged_results = old.get("results", [])
        print(f"   기존 문단 수: {len(merged_results)}")

    existing_contents = {item["content"] for item in merged_results}

    # JSON 파일 수집
    json_files = list(folder.glob("*.json"))
    print(f"\n📁 '{folder.name}' 폴더 내 JSON 파일 수: {len(json_files)}")

    new_items = []

    for file in json_files:
        print(f"   처리 중: {file.name}")
        data = load_json_file(file)
        if not data:
            continue

        category = extract_category_from_filename(file.name)

        # dict JSON
        if isinstance(data, dict):
            if "results" in data:
                for item in data["results"]:
                    if isinstance(item, dict) and "content" in item:
                        new_items.append({
                            "category": item.get("category", category),
                            "content": item["content"],
                            "source": file.name
                        })

            elif "content" in data:
                new_items.append({
                    "category": category,
                    "content": data["content"],
                    "source": file.name
                })

            else:
                new_items.append({
                    "category": category,
                    "content": json.dumps(data, ensure_ascii=False),
                    "source": file.name
                })

        # list JSON
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "content" in item:
                    new_items.append({
                        "category": item.get("category", category),
                        "content": item["content"],
                        "source": file.name
                    })
                else:
                    new_items.append({
                        "category": category,
                        "content": json.dumps(item, ensure_ascii=False),
                        "source": file.name
                    })

    print(f"✅ 새로 읽은 문단 수: {len(new_items)}")

    # 중복 제거 후 추가
    added_count = 0
    for item in new_items:
        if item["content"] not in existing_contents:
            merged_results.append(item)
            added_count += 1

    print(f"➕ 추가된 문단: {added_count}")
    print(f"📊 최종 문단 수: {len(merged_results)}")

    # 백업
    if os.path.exists(output_file):
        backup_file = output_file + ".backup"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump({"results": merged_results}, f, ensure_ascii=False, indent=2)
        print(f"💾 백업 생성: {backup_file}")

    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"results": merged_results}, f, ensure_ascii=False, indent=2)

    print(f"🎉 병합 완료 → {output_file}")


# ---------------------------------------------------------
# 스크립트 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE = "."

    TASKS = [
        ("data",      f"{BASE}/output/split_results_data.json"),
        ("sw_data",   f"{BASE}/output/split_results_sw_data.json"),
        ("swknu_data",f"{BASE}/output/split_results_swknu_data.json"),
    ]

    print("=" * 60)
    print("🔄 공주대학교 데이터 일괄 병합 스크립트")
    print("=" * 60)

    for folder, outfile in TASKS:
        print(f"\n=== 📌 폴더 병합: {folder} → {outfile}")
        merge_folder(f"{BASE}/{folder}", outfile)
