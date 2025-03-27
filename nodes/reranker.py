from typing import Dict, Any, List
import numpy as np
from core.base import Node
from llm.langchain_models import LangChainLLM, LangChainEmbeddings
from utils.logging import get_logger

logger = get_logger(__name__)


class SimpleReranker(Node):
    """Simple reranker node, reranks retrieval results based on similarity"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Initialize embedding model
        embedding_config = self.config.get("embedding", {})
        self.embedder = LangChainEmbeddings(embedding_config)
        
        # Number of documents to keep per collection
        self.top_k = self.config.get("top_k", 3)
        
        # Whether to apply Maximum Marginal Relevance (MMR) for diversification
        self.use_mmr = self.config.get("use_mmr", False)
        self.mmr_lambda = self.config.get("mmr_lambda", 0.5)  # MMR tradeoff parameter
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reranking logic"""
        query = state.get("query")
        documents = state.get("documents", [])
        
        if not query or not documents:
            logger.warning(f"Node {self.name}: No query provided or documents empty")
            return state
        
        try:
            # Calculate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # If documents don't have embeddings, recalculate
            contents = [doc.get("content", "") for doc in documents]
            if contents:
                embeddings = self.embedder.embed_documents(contents)
                
                # Update document embeddings
                for i, doc in enumerate(documents):
                    doc["embedding"] = embeddings[i]
            
            # Apply MMR or direct reranking
            if self.use_mmr and len(documents) > 1:
                reranked_docs = self._apply_mmr(documents, query_embedding)
            else:
                # Update similarity scores and rerank
                for doc in documents:
                    doc_embedding = doc.get("embedding")
                    if doc_embedding:
                        doc["score"] = self._cosine_similarity(query_embedding, doc_embedding)
                
                # Sort by score
                reranked_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
            
            # Limit document count
            reranked_docs = reranked_docs[:self.top_k]
            
            # Update state
            state["documents"] = reranked_docs
            logger.info(f"Node {self.name}: Kept {len(reranked_docs)} documents after reranking")
            
            return state
            
        except Exception as e:
            logger.error(f"Node {self.name} reranking failed: {e}")
            state["error"] = str(e)
            return state
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        a_array = np.array(a)
        b_array = np.array(b)
        
        dot_product = np.dot(a_array, b_array)
        norm_a = np.linalg.norm(a_array)
        norm_b = np.linalg.norm(b_array)
        
        return dot_product / (norm_a * norm_b)
    
    def _apply_mmr(self, documents: List[Dict[str, Any]], query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Apply Maximum Marginal Relevance (MMR) algorithm"""
        if len(documents) <= 1:
            return documents
        
        # Prepare document embeddings
        doc_embeddings = []
        for doc in documents:
            embedding = doc.get("embedding")
            if embedding:
                doc_embeddings.append(embedding)
            else:
                # If no embedding, use zero vector
                doc_embeddings.append([0.0] * len(query_embedding))
        
        # Calculate similarity of all docs to query
        query_similarities = [self._cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
        
        # Initialize selected and unselected documents
        selected_indices = []
        unselected_indices = list(range(len(documents)))
        
        # Select first most similar document
        first_idx = np.argmax(query_similarities)
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)
        
        # Iteratively select remaining documents
        while len(selected_indices) < len(documents) and unselected_indices:
            # Calculate MMR scores
            mmr_scores = []
            
            for i in unselected_indices:
                # Similarity to query
                sim_query = query_similarities[i]
                
                # Maximum similarity to selected docs
                max_sim_selected = max(
                    [self._cosine_similarity(doc_embeddings[i], doc_embeddings[j]) for j in selected_indices], 
                    default=0
                )
                
                # MMR formula: lambda * sim(q,d) - (1-lambda) * max_j sim(d,j)
                mmr_score = self.mmr_lambda * sim_query - (1 - self.mmr_lambda) * max_sim_selected
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            next_idx = unselected_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
            unselected_indices.remove(next_idx)
        
        # Reorder documents according to selection order
        reranked_docs = [documents[i] for i in selected_indices]
        
        # Update scores
        for i, doc in enumerate(reranked_docs):
            doc["score"] = query_similarities[selected_indices[i]]
        
        return reranked_docs


class LLMReranker(Node):
    """LLM-based advanced reranker"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Initialize LLM
        llm_config = self.config.get("llm", {})
        self.llm = LangChainLLM(llm_config)
        
        # Number of documents to keep per collection
        self.top_k = self.config.get("top_k", 3)
        
        # Get prompts from config
        self.system_prompt = self.config.get("system_prompt", "You are a professional information retrieval expert tasked with evaluating document relevance to a query.")
        self.prompt_template = self.config.get("prompt_template", "")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM reranking logic"""
        query = state.get("query")
        documents = state.get("documents", [])
        
        if not query or not documents:
            logger.warning(f"Node {self.name}: No query provided or documents empty")
            return state
        
        try:
            # Format document content
            docs_text = ""
            for i, doc in enumerate(documents):
                content = doc.get("content", "").strip()
                # Truncate overly long content to avoid token limit
                if len(content) > 500:
                    content = content[:500] + "..."
                docs_text += f"Document{i+1}:\n{content}\n\n"
            
            # Prepare prompt
            if self.prompt_template:
                prompt = self.prompt_template.format(
                    query=query,
                    documents=docs_text
                )
            else:
                prompt = (
                    f"Please evaluate the relevance of the following documents to the user query, and score each document (1-10 points).\n\n"
                    f"User query: {query}\n\n"
                    f"Documents:\n{docs_text}\n\n"
                    f"Please return in format:\nDocument1 score: [score]\nDocument2 score: [score]\n...\nReturn only scores, no explanations."
                )
            
            # Get scores
            response = self.llm.generate(prompt, system_message=self.system_prompt)
            
            # Parse scores
            scores = self._parse_scores(response, len(documents))
            
            # Update document scores
            for i, score in enumerate(scores):
                if i < len(documents):
                    documents[i]["score"] = score
            
            # Sort by score
            reranked_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
            
            # Limit document count
            reranked_docs = reranked_docs[:self.top_k]
            
            # Update state
            state["documents"] = reranked_docs
            logger.info(f"Node {self.name}: LLM reranking kept {len(reranked_docs)} documents")
            
            return state
            
        except Exception as e:
            logger.error(f"Node {self.name} LLM reranking failed: {e}")
            state["error"] = str(e)
            return state
    
    def _parse_scores(self, response: str, doc_count: int) -> List[float]:
        """Parse scores from LLM response"""
        scores = []
        
        # Use regex to match scores
        import re
        pattern = r"Document(\d+) score: *(\d+\.?\d*)"
        matches = re.findall(pattern, response)
        
        # Build score table, ensure correct order
        score_dict = {}
        for doc_idx_str, score_str in matches:
            try:
                doc_idx = int(doc_idx_str)
                score = float(score_str)
                score_dict[doc_idx] = score
            except ValueError:
                continue
        
        # Generate score list ordered by document index
        for i in range(1, doc_count + 1):
            if i in score_dict:
                scores.append(score_dict[i])
            else:
                # If a document has no score, use default
                scores.append(5.0)  # Medium relevance
        
        return scores