"""Multi LLMRouter RAG Workflow
   利用 LLMRouter 根據用戶查詢決定查詢 "NormalQA" 或 "HealthInsuranceFee"，再由相應檢索器查詢並由生成器產生回答。
"""

from core.component_manager import ComponentManager
from core.workflow_builder import WorkflowBuilder
from langgraph.graph import END
from nodes.custom import FunctionNode
from utils.logging import get_logger

logger = get_logger(__name__)

def build_workflow(component_manager):
    """
    建立工作流流程：
      1. 將輸入轉換為字典格式 (state_normalizer)
      2. 使用 LLMRouter (knowledge_router) 根據查詢決定 collection (結果存入 state["route"])
      3. 處理路由結果 (router_processor)
      4. 根據 state["route"] 選擇對應檢索器執行查詢 (retriever_selector)
      5. 生成器根據檢索結果生成回答 (answer_generator)
    """
    builder = WorkflowBuilder("multi_llm_router_workflow")
    
    # 1. 狀態標準化節點
    def ensure_state_dict(state):
        if isinstance(state, str):
            return {"query": state}
        return state
    builder.add_node("state_normalizer", FunctionNode("state_normalizer", ensure_state_dict))
    
    # 2. 使用 LLMRouter 進行路由，這裡會調用你提供的 LLMRouter 進行決策
    builder.add_node("knowledge_router", component_manager.create_router("knowledge_router"))
    
    # 3. 處理路由結果（僅作記錄，確保 state["route"] 正確）
    def process_router(state):
        route = state.get("route")
        logger.info(f"LLMRouter 選擇的 collection: {route}")
        return state
    builder.add_node("router_processor", FunctionNode("router_processor", process_router))
    
    # 4. 根據路由結果選擇對應的檢索器並執行查詢
    def select_retriever(state):
        route = state.get("route", "NormalQA")
        query = state.get("query", "")
        logger.info(f"根據路由結果選擇檢索器, route: {route}")
        retriever_map = {
            "NormalQA": "NormalQA",
            "HealthInsuranceFee": "HealthInsuranceFee"
        }
        retriever_name = retriever_map.get(route, "NormalQA")
        retriever = component_manager.create_retriever(retriever_name)
        retrieval_result = retriever({"query": query})
        logger.info(f"檢索結果: {retrieval_result}")
        state.update(retrieval_result)
        return state
    builder.add_node("retriever_selector", FunctionNode("retriever_selector", select_retriever))
    
    # 5. 生成器節點：根據檢索結果產生回答
    builder.add_node("answer_generator", component_manager.create_generator("default"))
    
    # 設置工作流入口
    builder.set_entry_point("state_normalizer")
    
    # 定義流程連接
    builder.add_edge("state_normalizer", "knowledge_router")
    builder.add_edge("knowledge_router", "router_processor")
    builder.add_edge("router_processor", "retriever_selector")
    builder.add_edge("retriever_selector", "answer_generator")
    builder.add_edge("answer_generator", END)
    
    return builder.compile()

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import logging
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    
    # 初始化組件管理器，讀取 YAML 配置文件
    component_manager = ComponentManager("configs/multi_llm_router_components.yaml")
    workflow = build_workflow(component_manager)
    
    # 測試查詢：例如查詢包含 "finance" 的問題
    test_state = {"query": "請問最新的 finance 資訊有哪些？"}
    result = workflow.invoke(test_state)
    
    print("最終回答：")
    print(result.get("answer", "未生成回答"))
