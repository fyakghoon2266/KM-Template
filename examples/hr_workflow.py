"""HR Agent知識庫工作流 - 修訂版"""
from core.component_manager import ComponentManager
from core.workflow_builder import WorkflowBuilder
from nodes.custom import FunctionNode
from langgraph.graph import END
from utils.logging import get_logger

logger = get_logger(__name__)

def build_workflow(component_manager):
    """
    建立HR Agent工作流 - 修訂版，解決狀態傳遞問題
    
    Args:
        component_manager: 組件管理器
        
    Returns:
        編譯好的工作流
    """
    # 創建工作流構建器
    builder = WorkflowBuilder("hr_agent_workflow")
    
    # 狀態處理函數
    def ensure_state_dict(state, default_key='query'):
        """確保狀態是字典格式"""
        logger.info(f"Processing state: {type(state).__name__} - {state}")
        if isinstance(state, str):
            return {default_key: state}
        return state
    
    # 路由結果處理函數 
    def process_route_result(state):
        """處理路由結果，確保返回正確的狀態對象"""
        logger.info(f"Processing route result: {state}")
        result = state.copy() if hasattr(state, 'copy') else {'query': state, 'route': state}
        return result
    
    # 添加狀態規範化節點
    builder.add_node(
        "state_normalizer", 
        FunctionNode("state_normalizer", lambda s: ensure_state_dict(s))
    )
    
    # 添加處理合法性檢查結果的節點
    builder.add_node(
        "legality_result_handler", 
        FunctionNode("legality_result_handler", process_route_result)
    )
    
    # 添加處理主題路由結果的節點
    builder.add_node(
        "topic_result_handler", 
        FunctionNode("topic_result_handler", process_route_result)
    )
    
    # 添加問題合法性檢查路由器
    builder.add_node(
        "question_legality_checker",
        component_manager.create_router("question_legality_checker")
    )
    
    # 添加QA節點
    builder.add_node(
        "qa_retriever",
        component_manager.create_retriever("NormalQA")
    )
    
    builder.add_node(
        "qa_processor",
        component_manager.create_generator("qa_processor")
    )
    
    # 添加不可回答問題處理節點
    builder.add_node(
        "cant_answer_retriever",
        component_manager.create_retriever("NormalQA")
    )
    
    builder.add_node(
        "cant_answer_handler",
        component_manager.create_generator("cant_answer_handler")
    )
    
    # 添加主題路由器
    builder.add_node(
        "topic_router",
        component_manager.create_router("hr_topic_router")
    )
    
    # 添加工具處理器
    builder.add_node(
        "tool_processor",
        component_manager.create_generator("tool_processor")
    )
    
    # 定義檢索器對應表
    retrievers = {
        "NormalQA": component_manager.create_retriever("NormalQA"),
        "WeddingsFuneralRelated": component_manager.create_retriever("WeddingsFuneralRelated"),
        "HealthInsuranceFee": component_manager.create_retriever("HealthInsuranceFee"),
        "HealthInsurance": component_manager.create_retriever("HealthInsurance"),
        "EmployeeStockowner": component_manager.create_retriever("EmployeeStockowner"),
        "Attendance": component_manager.create_retriever("Attendance"),
        "Volunteer": component_manager.create_retriever("Volunteer"),
        "Benefit": component_manager.create_retriever("Benefit"),
        "ContractEmployeeOff": component_manager.create_retriever("ContractEmployeeOff"),
        "Groupinsurance": component_manager.create_retriever("Groupinsurance"),
        "MedicalExamination": component_manager.create_retriever("MedicalExamination"),
        "PersonalChange": component_manager.create_retriever("PersonalChange"),
        "Oversea": component_manager.create_retriever("Oversea")
    }
    
    # 添加工具選擇器節點
    def select_retriever(state):
        """根據route選擇對應的檢索器"""
        logger.info(f"Selecting retriever based on state: {state}")
        route = state.get("route", "NormalQA")
        
        # 確保查詢參數正確傳遞
        query = state.get("query", "")
        if not query and "answer" in state:
            # 使用上一步生成的答案作為查詢
            query = state["answer"]
            
        # 使用選定的檢索器進行查詢
        selected_retriever = retrievers.get(route, retrievers["NormalQA"])
        retrieval_result = selected_retriever({"query": query})
        
        # 合併結果並返回
        result = state.copy()
        result.update(retrieval_result)
        return result
    
    builder.add_node(
        "tool_selector", 
        FunctionNode("tool_selector", select_retriever)
    )
    
    # 設置入口點並從標準化開始
    builder.set_entry_point("state_normalizer")
    
    # 設置工作流程
    builder.add_edge("state_normalizer", "question_legality_checker")
    builder.add_edge("question_legality_checker", "legality_result_handler")
    
    # 根據合法性結果分流
    builder.add_conditional_edge(
        "legality_result_handler",
        lambda state: state.get("route") == "continue",
        {True: "qa_retriever", False: "cant_answer_retriever"}
    )
    
    # 不可回答問題處理路徑
    builder.add_edge("cant_answer_retriever", "cant_answer_handler")
    builder.add_edge("cant_answer_handler", END)
    
    # QA檢索與處理路徑
    builder.add_edge("qa_retriever", "qa_processor")
    builder.add_edge("qa_processor", "topic_router")
    builder.add_edge("topic_router", "topic_result_handler")
    builder.add_edge("topic_result_handler", "tool_selector")
    builder.add_edge("tool_selector", "tool_processor")
    builder.add_edge("tool_processor", END)
    
    # 編譯並返回工作流
    return builder.compile()


if __name__ == "__main__":
    # 可以直接運行此文件進行測試
    from dotenv import load_dotenv
    import logging
    
    # 設置日誌級別以便調試
    logging.basicConfig(level=logging.INFO)
    
    # 加載環境變數
    load_dotenv()
    
    # 初始化組件管理器
    component_manager = ComponentManager("configs/hr_components.yaml")
    
    # 構建工作流
    workflow = build_workflow(component_manager)
    
    # 測試查詢 - 確保輸入格式正確
    print("\n===== 測試合法問題 =====")
    legal_result = workflow.invoke({"query": "我要請特休，請問流程是什麼?"})
    print(f"\n回答:\n{legal_result.get('answer', '未生成回答')}")
    
    # 測試查詢 - 不合法問題
    print("\n===== 測試不合法問題 =====")
    illegal_result = workflow.invoke({"query": "如何編寫Python程式來分析股票價格走勢?"})
    print(f"\n回答:\n{illegal_result.get('answer', '未生成回答')}")