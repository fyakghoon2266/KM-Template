from typing import Dict, Any, List, Optional
from langchain_qdrant import QdrantVectorStore as LCQdrantVectorStore
from qdrant_client import QdrantClient, models
from core.base import VectorStore
from utils.logging import get_logger

import os

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation, integrates LangChain's QdrantVectorStore"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qdrant client
        
        Args:
            config: Configuration dictionary
                - url: Qdrant server URL (required)
                - collections: Collection name list
                - vector_size: Vector dimension, defaults to 1536
                - distance: Distance metric, defaults to "Cosine"
        """
        self.url = os.environ['QDRANT_URL']
        if not self.url:
            raise ValueError("Must provide Qdrant URL")
            
        self.collections = config.get("collections", [])
        self.vector_size = config.get("vector_size", 1536)
        self.distance = config.get("distance", "Cosine")
        
        # Connect to Qdrant server
        self.client = QdrantClient(url=self.url)
        
        # Ensure collections exist
        self._ensure_collections()
        
        # Save embedding object, will be set later for querying
        self.embedding = None
        
        logger.info(f"Connected to Qdrant server: {self.url}")
    
    def _ensure_collections(self):
        """Ensure all collections exist"""
        for collection_name in self.collections:
            if not self.client.collection_exists(collection_name=collection_name):
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance[self.distance]
                    ),
                )
    
    def set_embedding(self, embedding):
        """Set embedding model"""
        self.embedding = embedding
    
    def _get_vector_store(self, collection_name: str):
        """Get LangChain vector store for specific collection"""
        # Check if collection exists
        if self.client.collection_exists(collection_name=collection_name):
            return LCQdrantVectorStore.from_existing_collection(
                embedding=self.embedding,
                collection_name=collection_name,
                url=self.url,
                content_payload_key='content',
                metadata_payload_key='metadata',
            )
        else:
            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance[self.distance]
                ),
            )
            return LCQdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embedding,
                content_payload_key='content',
                metadata_payload_key='metadata',
            )
    
    def query(self, 
              query_vector: List[float], 
              collection_name: str, 
              limit: int = 5,
              filter_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query for most similar documents"""
        # Create filter conditions
        search_kwargs = {"k": limit}
        if filter_params:
            must_filter_lst = [ 
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v),
                ) for k, v in filter_params.items()
            ]
            search_kwargs["filter"] = models.Filter(must=must_filter_lst)
        
        # Directly query using vector
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            **{k: v for k, v in search_kwargs.items() if k != 'k'}
        )
        
        # Convert results
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
                     collection_name: str) -> List[str]:
        """Add documents to vector store"""
        # Ensure collection exists
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance[self.distance]
                ),
            )
        
        # Prepare vector store
        vector_store = self._get_vector_store(collection_name)
        
        # Convert document format
        from langchain_core.documents import Document
        lc_docs = []
        for doc in documents:
            lc_docs.append(Document(
                page_content=doc["content"],
                metadata=doc.get("metadata", {})
            ))
        
        # Add documents
        ids = vector_store.add_documents(lc_docs)
        
        return ids
    
    def get_collections(self) -> List[str]:
        """Get all collection names"""
        collections_info = self.client.get_collections()
        return [c.name for c in collections_info.collections]