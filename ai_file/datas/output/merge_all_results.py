#!/usr/bin/env python3
"""
output 폴더의 3개 JSON 파일을 하나로 병합하는 스크립트
- split_results_data.json
- split_results_sw_data.json  
- split_results_swknu_data.json

→ merged_all_results.json
"""

import json
import os
from pathlib import Path

def merge_all_json_files(input_folder, output_file):
    """
    여러 JSON 파일의 results를 하나로 병합
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"❌ 입력 폴더 없음: {input_folder}")
        return
    
    # 병합할 파일 목록
    target_files = [
        "split_results_data.json",
        "split_results_sw_data.json",
        "split_results_swknu_data.json"
    ]
    
    merged_results = []
    total_items = 0
    
    print("=" * 70)
    print("🔄 공주대학교 데이터 통합 병합 스크립트")
    print("=" * 70)
    
    # 각 파일 처리
    for filename in target_files:
        filepath = input_path / filename
        
        if not filepath.exists():
            print(f"⚠️  파일 없음: {filename} (건너뜀)")
            continue
        
        print(f"\n📂 처리 중: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "results" in data:
                items = data["results"]
                count = len(items)
                merged_results.extend(items)
                total_items += count
                print(f"   ✅ {count:,}개 문단 추가")
                
            else:
                print(f"   ⚠️  'results' 키가 없음 (건너뜀)")
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"📊 최종 통계")
    print(f"{'=' * 70}")
    print(f"총 병합된 문단 수: {total_items:,}개")
    print(f"파일 처리 완료: {len([f for f in target_files if (input_path / f).exists()])}/{len(target_files)}개")
    
    # 결과 저장
    output_data = {
        "results": merged_results,
        "metadata": {
            "total_count": total_items,
            "source_files": target_files,
            "description": "공주대학교 통합 데이터 (data + sw_data + swknu_data)"
        }
    }
    
    # 출력 폴더 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 병합 완료!")
    print(f"📁 저장 위치: {output_file}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(output_file)
    size_mb = file_size / (1024 * 1024)
    print(f"📦 파일 크기: {size_mb:.2f} MB")
    
    return merged_results


if __name__ == "__main__":
    # 경로 설정 (스크립트가 output 폴더 안에 있음)
    INPUT_FOLDER = "."  # 현재 폴더 (output 폴더)
    OUTPUT_FILE = "./merged_all_results.json"  # 통합 결과 파일
    
    # 병합 실행
    merge_all_json_files(INPUT_FOLDER, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("✨ 작업 완료!")
    print("=" * 70)