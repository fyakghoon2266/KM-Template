from core.component_manager import ComponentManager
from core.workflow_builder import WorkflowBuilder

def build_workflow(component_manager: ComponentManager):
    builder = WorkflowBuilder("chatbox_workflow")
    
    # 直接建立生成器節點，將 CLI 輸入作為 prompt
    builder.add_node("generator", component_manager.create_generator("default"))
    
    # 設定入口節點為生成器
    builder.set_entry_point("generator")
    
    # 返回編譯好的工作流
    return builder.compile()