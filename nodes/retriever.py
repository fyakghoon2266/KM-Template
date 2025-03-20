from typing import Dict, Any, List, Optional

from core.base import Node
from vectorstores.qdrant_store import QdrantVectorStore
from llm.langchain_models import LangChainEmbeddings
from utils.logging import get_logger

logger = get_logger(__name__)


class QdrantRetriever(Node):
    """Qdrant retriever node"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Configure retrieval parameters
        self.collections = self.config.get("collections", [])
        if not self.collections:
            raise ValueError("Must specify at least one collection name")
        
        self.top_k = self.config.get("top_k", 3)
        self.filter_params = self.config.get("filter_params", None)
        
        # Initialize embedding model
        embedding_config = self.config.get("embedding", {})
        self.embedder = LangChainEmbeddings(embedding_config)
        
        # Initialize vector store
        qdrant_config = self.config.get("qdrant", {})
        qdrant_config["collections"] = self.collections
        self.vector_store = QdrantVectorStore(qdrant_config)
        
        # Set embedding model for vector store
        from langchain_openai import OpenAIEmbeddings
        
        # Extract configuration
        model = embedding_config.get("model", "text-embedding-ada-002")
        api_key = embedding_config.get("api_key")
        api_base = embedding_config.get("api_base")
        
        # Create embedding object
        kwargs = {}
        if api_key:
            kwargs["openai_api_key"] = api_key
        if api_base:
            kwargs["openai_api_base"] = api_base
        
        try:
            openai_embedding = OpenAIEmbeddings(
                model=model,
                **kwargs
            )
            self.vector_store.set_embedding(openai_embedding)
        except Exception as e:
            logger.error(f"OpenAI embedding initialization failed: {e}")
            # Can set up a mock embedding as fallback
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieval logic"""
        # Get query text
        query = state.get("query")
        if not query:
            logger.warning(f"Node {self.name}: No query text provided")
            state["documents"] = []
            return state
        
        try:
            # Embed query
            query_vector = self.embedder.embed_query(query)
            
            # Store all retrieval results
            all_docs = []
            
            # Retrieve documents from each collection
            for collection in self.collections:
                docs = self.vector_store.query(
                    query_vector=query_vector,
                    collection_name=collection,
                    limit=self.top_k,
                    filter_params=self.filter_params
                )
                
                # Add collection source
                for doc in docs:
                    doc["source_collection"] = collection
                
                all_docs.extend(docs)
            
            # If multiple collections, sort by relevance
            if len(self.collections) > 1:
                all_docs.sort(key=lambda x: x["score"], reverse=True)
                # Keep only top_k documents
                all_docs = all_docs[:self.top_k]
            
            # Update state
            state["documents"] = all_docs
            state["retriever_collections"] = self.collections
            
            logger.info(f"Node {self.name}: Retrieved {len(all_docs)} documents")
            
            return state
            
        except Exception as e:
            logger.error(f"Node {self.name} retrieval failed: {e}")
            state["error"] = str(e)
            state["documents"] = []
            return state


class MultiQdrantRetriever(QdrantRetriever):
    """Multi-Qdrant collection retriever node, supports advanced multi-RAG strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Collection weight configuration
        self.collection_weights = self.config.get("collection_weights", {})
        
        # Mixing strategy: 'interleave', 'weighted', 'ensemble'
        self.mixing_strategy = self.config.get("mixing_strategy", "weighted")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-collection retrieval logic"""
        # Get query text
        query = state.get("query")
        if not query:
            logger.warning(f"Node {self.name}: No query text provided")
            state["documents"] = []
            return state
        
        try:
            # Embed query
            query_vector = self.embedder.embed_query(query)
            
            # Retrieve from each collection
            collection_docs = {}
            for collection in self.collections:
                docs = self.vector_store.query(
                    query_vector=query_vector,
                    collection_name=collection,
                    limit=self.top_k * 2,  # Retrieve more for subsequent mixing
                    filter_params=self.filter_params
                )
                
                # Add collection source
                for doc in docs:
                    doc["source_collection"] = collection
                
                collection_docs[collection] = docs
            
            # Mix results according to strategy
            if self.mixing_strategy == "interleave":
                all_docs = self._interleave_results(collection_docs)
            elif self.mixing_strategy == "weighted":
                all_docs = self._weighted_results(collection_docs)
            elif self.mixing_strategy == "ensemble":
                all_docs = self._ensemble_results(collection_docs)
            else:
                # Default simple merge
                all_docs = []
                for docs in collection_docs.values():
                    all_docs.extend(docs)
                all_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Keep only top_k documents
            all_docs = all_docs[:self.top_k]
            
            # Update state
            state["documents"] = all_docs
            state["retriever_collections"] = self.collections
            state["collection_docs"] = collection_docs
            
            logger.info(f"Node {self.name}: Retrieved {len(all_docs)} documents using {self.mixing_strategy} strategy")
            
            return state
            
        except Exception as e:
            logger.error(f"Node {self.name} retrieval failed: {e}")
            state["error"] = str(e)
            state["documents"] = []
            return state
    
    def _interleave_results(self, collection_docs: Dict[str, List[Dict]]) -> List[Dict]:
        """Interleave mixing results"""
        all_docs = []
        max_docs = max(len(docs) for docs in collection_docs.values())
        
        for i in range(max_docs):
            for collection in self.collections:
                docs = collection_docs[collection]
                if i < len(docs):
                    all_docs.append(docs[i])
        
        return all_docs
    
    def _weighted_results(self, collection_docs: Dict[str, List[Dict]]) -> List[Dict]:
        """Weighted mixing results"""
        all_docs = []
        
        for collection in self.collections:
            docs = collection_docs[collection]
            weight = self.collection_weights.get(collection, 1.0)
            
            for doc in docs:
                # Apply weight
                doc["score"] = doc["score"] * weight
                all_docs.append(doc)
        
        # Sort by adjusted score
        all_docs.sort(key=lambda x: x["score"], reverse=True)
        return all_docs
    
    def _ensemble_results(self, collection_docs: Dict[str, List[Dict]]) -> List[Dict]:
        """Ensemble mixing results, using weighted voting"""
        all_docs = []
        doc_scores = {}
        
        # Collect all documents
        for collection, docs in collection_docs.items():
            weight = self.collection_weights.get(collection, 1.0)
            for doc in docs:
                doc_id = doc["id"]
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "scores": []}
                
                # Record weighted score
                doc_scores[doc_id]["scores"].append(doc["score"] * weight)
        
        # Calculate merged scores
        for doc_id, data in doc_scores.items():
            if len(data["scores"]) > 0:
                # Use maximum score as final score
                data["doc"]["score"] = max(data["scores"])
                all_docs.append(data["doc"])
        
        # Sort by final score
        all_docs.sort(key=lambda x: x["score"], reverse=True)
        return all_docs