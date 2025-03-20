# LangGraph RAG Framework

LangGraph RAG Framework是一個基於LangGraph構建的知識庫框架，讓使用者能夠輕鬆創建、自訂和部署RAG應用。

## 框架特點

- **組件配置**：通過YAML文件配置組件參數
- **工作流構建**：使用Python API靈活構建工作流
- **多種檢索策略**：支持多種向量檢索和混合策略
- **自訂節點**：輕鬆擴展框架功能
- **分布式設計**：可擴展到大規模應用

## 快速開始

### 安裝

```bash
pip install -r requirements.txt
配置環境變數
創建 .env 文件並設置必要的API密鑰：

OPENAI_API_KEY='your_openai_api_key'
OPENAI_ENDPOINT='your_openai_endpoint'
OPENAI_DEPLOYMENT='your_openai_deployment'
OPENAI_EMBEDDING_DEPLOYMENT='your_embedding_deployment'
QDRANT_URL='your_qdrant_url'
運行示例
# 使用簡單RAG工作流
python main.py --config configs/components.yaml --workflow examples/simple_rag_workflow.py --query "什麼是向量數據庫?"

# 使用多知識庫工作流
python main.py --config configs/components.yaml --workflow examples/multi_collection_workflow.py --query "我要怎麼請特休?"
創建自定義工作流
1. 創建組件配置文件
在 configs/ 目錄下創建YAML配置文件，定義所需組件：

# 檢索器配置
retrievers:
  default:
    top_k: 5
    qdrant:
      url: \${QDRANT_URL}
    embedding:
      model: \${OPENAI_EMBEDDING_DEPLOYMENT}

# 生成器配置
generators:
  default:
    llm:
      model: \${OPENAI_DEPLOYMENT}
      temperature: 0.7
2. 創建工作流Python文件
from core.component_manager import ComponentManager
from core.workflow_builder import WorkflowBuilder

def build_workflow(component_manager):
    # 創建工作流構建器
    builder = WorkflowBuilder("my_workflow")
    
    # 添加節點
    retriever = builder.add_node("retriever", 
                               component_manager.create_retriever("default", 
                                                               collections=["my_collection"]))
    generator = builder.add_node("generator", 
                               component_manager.create_generator("default"))
    
    # 設置入口點
    builder.set_entry_point("retriever")
    
    # 添加邊
    builder.add_edge("retriever", "generator")
    
    # 編譯並返回工作流
    return builder.compile()
3. 運行工作流
python main.py --config configs/components.yaml --workflow path/to/my_workflow.py --query "您的問題"
框架架構
configs/: 組件配置文件
core/: 核心框架邏輯
examples/: 示例工作流
llm/: 語言模型包裝器
nodes/: 節點實現
utils/: 工具函數
vectorstores/: 向量存儲實現