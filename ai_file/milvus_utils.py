# milvus_utils.py
"""
Milvus 벡터 DB 연결 및 검색 유틸리티
"""
from pymilvus import connections, Collection
from config import MILVUS_CONFIG
import numpy as np

class MilvusClient:
    def __init__(self):
        self.collection = None
        self.connected = False
    
    def connect(self):
        """Milvus 서버 연결"""
        try:
            connections.connect(
                alias="default",
                host=MILVUS_CONFIG['host'],
                port=MILVUS_CONFIG['port']
            )
            self.collection = Collection(MILVUS_CONFIG['collection_name'])
            self.collection.load()  # 컬렉션 메모리 로드 (중요!)
            self.connected = True
            print(f"[Milvus] 연결 성공: {MILVUS_CONFIG['collection_name']}")
            print(f"[Milvus] 벡터 개수: {self.collection.num_entities}")
        except Exception as e:
            print(f"[Milvus] 연결 실패: {e}")
            self.connected = False
            raise
    
    def search(self, query_vector, top_k=3):
        """
        벡터 검색
        
        Args:
            query_vector: numpy array (384,)
            top_k: 반환할 결과 개수
        
        Returns:
            list of dict: [{"text": ..., "score": ...}]
        """
        if not self.connected:
            raise ConnectionError("Milvus에 연결되지 않았습니다.")
        
        # numpy array를 list로 변환
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        # 검색 파라미터
        search_params = {
            "metric_type": MILVUS_CONFIG['metric_type'],
            "params": {"nprobe": 10}
        }
        
        # 검색 실행
        results = self.collection.search(
            data=[query_vector],
            anns_field=MILVUS_CONFIG['vector_field'],  # 'embedding'
            param=search_params,
            limit=top_k,
            output_fields=["id", "text"]  # ← 변경! (menu, url, content → id, text)
        )
        
        # 결과 변환
        matched = []
        for hit in results[0]:
            matched.append({
                "content": hit.entity.get("text", ""),  # text → content로 매핑
                "category": "",  # 없으면 빈 문자열
                "score": float(hit.score),
                "id": hit.entity.get("id", "")
            })
        
        return matched
    
    def disconnect(self):
        """연결 종료"""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
            print("[Milvus] 연결 종료")

# 싱글톤 인스턴스
_milvus_client = None

def get_milvus_client():
    """Milvus 클라이언트 싱글톤"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
        try:
            _milvus_client.connect()
        except Exception as e:
            print(f"[경고] Milvus 연결 실패: {e}")
            print(f"[경고] FAISS 모드로 폴백합니다")
            return None  # ← None 반환하면 ai_server.py에서 처리
    return _milvus_client