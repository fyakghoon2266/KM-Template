from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import importlib.util
import sys
import os

from core.component_manager import ComponentManager
from utils.logging import get_logger, setup_file_logging

app = FastAPI()
logger = get_logger(__name__)

# 預加載工作流
workflows = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    workflow: str  # 工作流名稱
    history: Optional[List[ChatMessage]] = None
    parameters: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    history: List[ChatMessage]
    sources: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """處理聊天請求"""
    workflow_name = request.workflow
    
    # 檢查工作流是否已加載
    if workflow_name not in workflows:
        try:
            # 動態加載工作流
            workflow_path = f"examples/{workflow_name}.py"
            spec = importlib.util.spec_from_file_location(f"workflow_{workflow_name}", workflow_path)
            if not spec or not spec.loader:
                raise HTTPException(status_code=404, detail=f"工作流 {workflow_name} 不存在")
                
            workflow_module = importlib.util.module_from_spec(spec)
            sys.modules[f"workflow_{workflow_name}"] = workflow_module
            spec.loader.exec_module(workflow_module)
            
            # 檢查模塊中是否有build_workflow函數
            if not hasattr(workflow_module, "build_workflow"):
                raise HTTPException(status_code=500, detail=f"工作流 {workflow_name} 中沒有定義 build_workflow 函數")
            
            # 構建工作流
            component_manager = ComponentManager("configs/components.yaml")
            workflow = workflow_module.build_workflow(component_manager)
            workflows[workflow_name] = workflow
            
        except Exception as e:
            logger.error(f"加載工作流失敗: {e}")
            raise HTTPException(status_code=500, detail=f"加載工作流失敗: {str(e)}")
    
    try:
        # 執行查詢
        result = workflows[workflow_name].invoke({
            "query": request.query,
            "history": [dict(msg) for msg in (request.history or [])],
            **(request.parameters or {})
        })
        
        # 構建響應
        response = {
            "answer": result.get("answer", "未生成回答"),
            "history": result.get("history", []),
            "sources": result.get("sources", [])
        }
        
        if "error" in result:
            response["error"] = result["error"]
        
        return response
        
    except Exception as e:
        logger.error(f"處理查詢失敗: {e}")
        raise HTTPException(status_code=500, detail=f"處理查詢失敗: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # 設置日誌
    setup_file_logging("logs")
    
    # 運行API服務器
    uvicorn.run(app, host="0.0.0.0", port=8000)