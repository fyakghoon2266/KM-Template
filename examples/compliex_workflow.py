from core.component_manager import ComponentManager
from core.workflow_builder import WorkflowBuilder


def build_workflow(component_manager):
    """建立包含合法性檢查的工作流，簡化版"""
    
    # 創建工作流構建器
    builder = WorkflowBuilder("complex_workflow")
    
    # 添加節點
    # 1. 合法性檢查路由器
    builder.add_node(
        "legality_checker",
        component_manager.create_router("legality_checker")
    )
    
    # 2. 正常問題路由器 (當問題合法時使用)
    builder.add_node(
        "category_router",
        component_manager.create_router("bank_router")
    )
    
    # 3. 只使用實際有的檢索器
    builder.add_node(
        "normal_retriever",
        component_manager.create_retriever("normal_qa")
    )
    
    builder.add_node(
        "wedding_retriever",
        component_manager.create_retriever("wedding_funeral")
    )
    
    # 4. 生成器
    builder.add_node(
        "normal_generator",
        component_manager.create_generator("bank_assistant")
    )
    
    # 5. 拒絕處理器
    builder.add_node(
        "rejection_handler",
        component_manager.create_generator("rejection_handler")
    )
    
    # 設置入口點
    builder.set_entry_point("legality_checker")
    
    # 添加條件邊 - 合法性檢查後的路由
    builder.add_conditional_edge(
        "legality_checker",
        lambda state: state.get("route") == "legal",
        {True: "category_router", False: "rejection_handler"}
    )
    
    # 添加條件邊 - 分類路由
    builder.add_conditional_edge(
        "category_router",
        lambda state: state.get("route") == "normal_qa",
        {True: "normal_retriever", False: "wedding_retriever"}  # 簡化路由邏輯
    )
    
    # 添加到生成器的邊
    builder.add_edge("normal_retriever", "normal_generator")
    builder.add_edge("wedding_retriever", "normal_generator")
    
    # 編譯並返回工作流
    return builder.compile()


if __name__ == "__main__":
    # 可以直接運行此文件進行測試
    import os
    from dotenv import load_dotenv
    
    # 加載環境變數
    load_dotenv()
    
    # 初始化組件管理器
    component_manager = ComponentManager("configs/components.yaml")
    
    # 構建工作流
    workflow = build_workflow(component_manager)
    
    # 測試查詢 - 合法問題
    legal_result = workflow.invoke({"query": "我要怎麼請特休?"})
    print(f"\n合法問題回答:\n{legal_result.get('answer', '未生成回答')}")
    
    # 測試查詢 - 不合法問題
    illegal_result = workflow.invoke({"query": "如何製作一個機械鍵盤?"})
    print(f"\n不合法問題回答:\n{illegal_result.get('answer', '未生成回答')}")