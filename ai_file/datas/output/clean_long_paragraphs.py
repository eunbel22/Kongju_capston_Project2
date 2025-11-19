#!/usr/bin/env python3
"""
merged_all_results.json에서 너무 긴 문단을 정리하는 스크립트
"""

import json
import os

def clean_long_paragraphs(input_file, output_file, max_length=5000):
    """
    너무 긴 문단을 필터링하거나 자르기
    
    Args:
        input_file: 입력 JSON 파일
        output_file: 출력 JSON 파일
        max_length: 최대 문자 길이 (기본 5000자)
    """
    print(f"📂 입력 파일: {input_file}")
    
    # 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    print(f"📊 원본 문단 수: {len(results):,}개")
    
    # 통계
    total = len(results)
    filtered = []
    truncated_count = 0
    removed_count = 0
    
    for item in results:
        content = item.get("content", "")
        content_len = len(content)
        
        if content_len > max_length * 2:
            # 너무 긴 문단은 제거
            removed_count += 1
            print(f"  ❌ 제거: {item.get('category', 'Unknown')[:50]} (길이: {content_len:,}자)")
            continue
        elif content_len > max_length:
            # 적당히 긴 문단은 자르기
            item["content"] = content[:max_length] + "...(이하 생략)"
            truncated_count += 1
            filtered.append(item)
        else:
            # 정상 길이
            filtered.append(item)
    
    print(f"\n📊 처리 결과:")
    print(f"   - 원본: {total:,}개")
    print(f"   - 유지: {len(filtered):,}개")
    print(f"   - 잘림: {truncated_count:,}개")
    print(f"   - 제거: {removed_count:,}개")
    
    # 저장
    output_data = {
        "results": filtered,
        "metadata": {
            "total_count": len(filtered),
            "max_length": max_length,
            "truncated": truncated_count,
            "removed": removed_count,
            "description": "길이 제한 적용된 공주대학교 통합 데이터"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 저장 완료: {output_file}")
    
    # 파일 크기
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"📦 파일 크기: {file_size:.2f} MB")


if __name__ == "__main__":
    INPUT = "./merged_all_results.json"
    OUTPUT = "./merged_all_results_cleaned.json"
    
    print("=" * 70)
    print("🧹 긴 문단 정리 스크립트")
    print("=" * 70)
    
    if not os.path.exists(INPUT):
        print(f"❌ 파일 없음: {INPUT}")
    else:
        clean_long_paragraphs(INPUT, OUTPUT, max_length=5000)
        
        print("\n" + "=" * 70)
        print("✨ 완료!")
        print("=" * 70)
        print("\n💡 다음 단계:")
        print("1. config.py에서 DATA_FILES를 'merged_all_results_cleaned.json'으로 변경")
        print("2. regenerate_embeddings.py 실행")