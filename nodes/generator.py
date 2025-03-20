from typing import Dict, Any, List, Optional
from core.base import Node
from llm.langchain_models import LangChainLLM
from utils.logging import get_logger

logger = get_logger(__name__)


class LLMGenerator(Node):
    """基於LLM的生成節點"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # 初始化LLM
        llm_config = self.config.get("llm", {})
        self.llm = LangChainLLM(llm_config)
        
        # 從配置中獲取系統提示和用戶提示模板
        self.system_prompt = self.config.get("system_prompt", "")
        self.prompt_template = self.config.get("prompt_template", "{query}\n\n{context}")
        
        # 是否啟用源引用
        self.include_sources = self.config.get("include_sources", True)
        
        # 是否包含歷史對話
        self.include_history = self.config.get("include_history", True)
        self.max_history_turns = self.config.get("max_history_turns", 5)

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
    
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """格式化歷史對話，更加健壯"""
        # 確保歷史是有效列表
        if not history or not isinstance(history, list):
            return ""
        
        # 過濾無效條目
        valid_history = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                valid_history.append((role, content))
        
        # 如果沒有有效歷史，直接返回空
        if not valid_history:
            return ""
        
        # 取最近的n輪對話
        recent_history = valid_history[-self.max_history_turns:]
        
        formatted_history = "先前對話:\n"
        for role, content in recent_history:
            if role == "user":
                formatted_history += f"用戶: {content}\n"
            elif role == "assistant":
                formatted_history += f"助手: {content}\n"
        
        return formatted_history + "\n"
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 修改這部分確保更健壯
        query = state.get("query", "")
        documents = state.get("documents", [])
        
        # 確保歷史始終是有效列表
        history = state.get("history")
        if not isinstance(history, list):
            history = []
        
        if not query:
            logger.warning(f"節點 {self.name}: 未提供查詢文本")
            state["answer"] = "無法生成回答，因為沒有提供問題。"
            return state
        
        try:
            # 格式化上下文和歷史
            context = self._format_documents(documents)
            history_text = self._format_history(history) if self.include_history and history else ""
        
            # 準備提示，根據history_text是否為空決定如何格式化
            if history_text:
                prompt = self.prompt_template.format(
                    query=query,
                    context=context,
                    history=history_text
                )
            else:
                # 如果沒有歷史，使用簡化版提示模板
                simple_template = self.prompt_template.replace("{history}\n\n", "").replace("{history}", "")
                prompt = simple_template.format(
                    query=query,
                    context=context
                )
            
            # 生成回答
            answer = self.llm.generate(prompt, system_message=self.system_prompt)
            
            # 更新狀態
            state["answer"] = answer
            
            # 添加新的對話記錄
            if "history" not in state:
                state["history"] = []
            
            # 添加當前問題和回答到歷史記錄
            state["history"].append({"role": "user", "content": query})
            state["history"].append({"role": "assistant", "content": answer})
            
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