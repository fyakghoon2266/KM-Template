"""Qdrant向量存储实现"""
from typing import Dict, Any, List, Optional
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from core.base import VectorStore
from core.types import Document
from utils.logging import get_logger

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant向量存储实现"""
    
    def __init__(self, 
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 path: Optional[str] = None,
                 vector_size: int = 1536):
        """
        初始化Qdrant客户端
        
        Args:
            url: Qdrant服务器URL
            api_key: Qdrant API密钥
            path: 本地Qdrant路径
            vector_size: 向量维度
        """
        self.vector_size = vector_size
        
        # 优先使用环境变量
        url = url or os.environ.get("QDRANT_URL")
        api_key = api_key or os.environ.get("QDRANT_API_KEY")
        path = path or os.environ.get("QDRANT_PATH")
        
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"已连接到远程Qdrant服务器: {url}")
        elif path:
            self.client = QdrantClient(path=path)
            logger.info(f"已连接到本地Qdrant: {path}")
        else:
            # 默认使用内存模式
            self.client = QdrantClient()
            logger.info("使用内存模式的Qdrant客户端")
        
        self._ensure_collections()
    
    def _ensure_collections(self, collection_names: List[str] = None):
        """确保集合存在"""
        if not collection_names:
            return
            
        collections_info = self.client.get_collections()
        existing_collections = [c.name for c in collections_info.collections]
        
        for name in collection_names:
            if name not in existing_collections:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                logger.info(f"已创建集合: {name}")
    
    def query(self, 
              query_vector: List[float], 
              collection_name: str, 
              limit: int = 5,
              filter_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """查询最相似的文档"""
        # 构建筛选条件
        filter_obj = None
        if filter_params:
            conditions = []
            for field, value in filter_params.items():
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
            filter_obj = Filter(must=conditions)
        
        # 执行查询
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            filter=filter_obj
        )
        
        # 转换结果
        documents = []
        for result in results:
            doc = {
                "id": result.id,
                "content": result.payload.get("content", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "content"},
                "score": result.score
            }
            documents.append(doc)
        
        return documents
    
    def add_documents(self, 
                     documents: List[Dict[str, Any]], 
                     collection_name: str,
                     vectors: List[List[float]]) -> List[str]:
        """添加文档到向量存储"""
        # 确保集合存在
        self._ensure_collections([collection_name])
        
        # 构建点结构
        points = []
        ids = []
        
        for i, (doc, vec) in enumerate(zip(documents, vectors)):
            doc_id = doc.get("id") or str(i)
            ids.append(doc_id)
            
            # 构建payload
            payload = {"content": doc["content"]}
            if "metadata" in doc:
                payload.update(doc["metadata"])
            
            # 创建点
            points.append(PointStruct(
                id=doc_id,
                vector=vec,
                payload=payload
            ))
        
        # 批量插入
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return ids
    
    def get_collections(self) -> List[str]:
        """获取所有集合名称"""
        collections_info = self.client.get_collections()
        return [c.name for c in collections_info.collections]