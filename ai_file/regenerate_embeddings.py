#!/usr/bin/env python3
"""
merged_all_results.json 파일로 임베딩과 FAISS 인덱스 재생성
"""

import os
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_FILES, EMBED_PATH, INDEX_PATH
from data_utils import load_paragraphs, prepare_faiss
from embedding_utils import load_embed_model

def regenerate_embeddings():
    """
    통합된 JSON 파일로 임베딩과 FAISS 인덱스를 재생성
    """
    print("=" * 70)
    print("🔄 임베딩 & FAISS 인덱스 재생성")
    print("=" * 70)
    
    # 1. 통합 JSON 파일 로드
    print(f"\n📂 데이터 파일: {DATA_FILES[0]}")
    
    if not os.path.exists(DATA_FILES[0]):
        print(f"❌ 파일이 존재하지 않습니다: {DATA_FILES[0]}")
        print("먼저 merge_all_results.py를 실행하여 통합 파일을 생성하세요!")
        return False
    
    paragraphs = load_paragraphs(DATA_FILES[0])
    print(f"✅ 로드된 문단 수: {len(paragraphs):,}개")
    
    if len(paragraphs) == 0:
        print("❌ 문단 데이터가 없습니다!")
        return False
    
    # 2. 임베딩 모델 로드
    print("\n🤖 임베딩 모델 로드 중...")
    tokenizer, model = load_embed_model()
    print("✅ 모델 로드 완료")
    
    # 3. 임베딩 & FAISS 인덱스 생성
    print("\n⚙️  임베딩 생성 중... (시간이 걸릴 수 있습니다)")
    
    # 기존 파일 삭제 (강제 재생성)
    if os.path.exists(EMBED_PATH):
        os.remove(EMBED_PATH)
        print(f"🗑️  기존 임베딩 파일 삭제: {EMBED_PATH}")
    
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        print(f"🗑️  기존 인덱스 파일 삭제: {INDEX_PATH}")
    
    # models 폴더 생성
    os.makedirs(os.path.dirname(EMBED_PATH), exist_ok=True)
    
    try:
        index = prepare_faiss(
            paragraphs=paragraphs,
            json_path=DATA_FILES[0],
            embed_path=EMBED_PATH,
            index_path=INDEX_PATH,
            tokenizer=tokenizer,
            model=model
        )
        
        print("\n✅ 임베딩 & 인덱스 생성 완료!")
        print(f"📁 임베딩 저장: {EMBED_PATH}")
        print(f"📁 인덱스 저장: {INDEX_PATH}")
        
        # 파일 크기 확인
        embed_size = os.path.getsize(EMBED_PATH) / (1024 * 1024)
        index_size = os.path.getsize(INDEX_PATH) / (1024 * 1024)
        print(f"\n📊 파일 크기:")
        print(f"   - embeddings.npy: {embed_size:.2f} MB")
        print(f"   - faiss.index: {index_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 공주대학교 통합 데이터 임베딩 재생성")
    print("=" * 70)
    
    success = regenerate_embeddings()
    
    print("\n" + "=" * 70)
    if success:
        print("✨ 작업 완료! 이제 서버를 재시작하세요.")
    else:
        print("❌ 작업 실패!")
    print("=" * 70)