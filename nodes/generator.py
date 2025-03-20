from typing import Dict, Any, List, Optional
from core.base import Node
from llm.langchain_models import LangChainLLM
from utils.logging import get_logger

logger = get_logger(__name__)


class LLMGenerator(Node):
    """LLM-based generator node"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Initialize LLM
        llm_config = self.config.get("llm", {})
        self.llm = LangChainLLM(llm_config)
        
        # Get system prompt and user prompt template from config
        self.system_prompt = self.config.get("system_prompt", "")
        self.prompt_template = self.config.get("prompt_template", "{query}\n\n{context}")
        
        # Whether to enable source citations
        self.include_sources = self.config.get("include_sources", True)
    
    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents as context string"""
        if not documents:
            return "No reference information available."
        
        context_parts = []
        for i, doc in enumerate(documents):
            source = f"[Source: {doc.get('source_collection', 'Unknown')}]" if self.include_sources else ""
            content = doc.get("content", "")
            context_parts.append(f"Document {i+1} {source}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generation logic"""
        query = state.get("query")
        documents = state.get("documents", [])
        
        if not query:
            logger.warning(f"Node {self.name}: No query text provided")
            state["answer"] = "Cannot generate answer because no question was provided."
            return state
        
        try:
            # Format context
            context = self._format_documents(documents)
            
            # Prepare prompt
            prompt = self.prompt_template.format(
                query=query,
                context=context
            )
            
            # Generate answer
            answer = self.llm.generate(prompt, system_message=self.system_prompt)
            
            # Update state
            state["answer"] = answer
            
            # Save source citations
            if self.include_sources:
                sources = []
                for doc in documents:
                    if "source_collection" in doc:
                        if "metadata" in doc:
                            source = {"collection": doc["source_collection"], "metadata": doc["metadata"]}
                        else:
                            source = {"collection": doc["source_collection"]}
                        sources.append(source)
                state["sources"] = sources
            
            logger.info(f"Node {self.name}: Answer generated")
            
            return state
            
        except Exception as e:
            logger.error(f"Node {self.name} generation failed: {e}")
            state["error"] = str(e)
            state["answer"] = "An error occurred while generating the answer."
            return state