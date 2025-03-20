from core.component_manager import ComponentManager
from core.workflow_builder import WorkflowBuilder
from core.base import Node


class RouterWrapper(Node):
    """路由器包裝節點，集成路由和檢索邏輯"""
    
    def __init__(self, name: str, config=None):
        super().__init__(name, config or {})
        self.llm_router = None
        self.normal_retriever = None
        self.wedding_retriever = None
    
    def __call__(self, state):
        # First run the LLM router
        if self.llm_router:
            state = self.llm_router(state)
        
        # Get the route decision
        route = state.get("route")
        
        # Run the appropriate retriever based on route
        if route == "normal_qa" and self.normal_retriever:
            return self.normal_retriever(state)
        elif route == "wedding_funeral" and self.wedding_retriever:
            return self.wedding_retriever(state)
        else:
            # Default fallback to normal retriever
            if self.normal_retriever:
                return self.normal_retriever(state)
            return state


def build_workflow(component_manager):
    """
    構建多知識庫工作流
    
    Args:
        component_manager: 組件管理器
        
    Returns:
        編譯好的工作流
    """
    # 創建工作流構建器
    builder = WorkflowBuilder("multi_collection_workflow")
    
    # 創建檢索器
    normal_retriever = component_manager.create_retriever("normal_qa")
    wedding_retriever = component_manager.create_retriever("wedding_funeral")
    
    # 創建並配置路由包裝器
    router_wrapper = RouterWrapper("router")
    router_wrapper.llm_router = component_manager.create_router("bank_router")
    router_wrapper.normal_retriever = normal_retriever
    router_wrapper.wedding_retriever = wedding_retriever
    
    # 添加節點
    builder.add_node("router", router_wrapper)
    builder.add_node("generator", component_manager.create_generator("bank_assistant"))
    
    # 設置入口點
    builder.set_entry_point("router")
    
    # 從路由器直接連接到生成器
    builder.add_edge("router", "generator")
    
    # 編譯並返回工作流
    return builder.compile()

if __name__ == "__main__":
    # Can run this file directly for testing
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Initialize component manager
    component_manager = ComponentManager("configs/components.yaml")
    
    # Build workflow
    workflow = build_workflow(component_manager)
    
    # Test query
    result = workflow.invoke({"query": "我要怎麼請特休?"})
    # print(f"\nAnswer:\n{result.get('answer', '未生成回答')}")

    print(result)
