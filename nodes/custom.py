from typing import Dict, Any, Callable, List, Optional
import inspect
from core.base import Node
from utils.logging import get_logger

logger = get_logger(__name__)


class FunctionNode(Node):
    """函数节点，可将普通函数包装为节点"""
    
    def __init__(self, name: str, func: Callable, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.func = func
        self._validate_function()
    
    def _validate_function(self):
        """验证函数是否具有正确签名"""
        sig = inspect.signature(self.func)
        if len(sig.parameters) != 1:
            logger.warning(f"函数节点 {self.name}: 函数应该只接受一个状态参数")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行函数"""
        try:
            return self.func(state)
        except Exception as e:
            logger.error(f"函数节点 {self.name} 执行失败: {e}")
            state["error"] = str(e)
            return state


class CustomNode(Node):
    """自定义节点基类，便于用户扩展"""
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理状态，用户应重写此方法实现自定义逻辑
        
        Args:
            state: 工作流状态
        
        Returns:
            更新后的状态
        """
        raise NotImplementedError("子类必须实现process方法")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行节点逻辑"""
        try:
            return self.process(state)
        except Exception as e:
            logger.error(f"自定义节点 {self.name} 执行失败: {e}")
            state["error"] = str(e)
            return state


class ConditionalNode(CustomNode):
    """条件节点，根据条件决定是否执行处理"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.condition = self.config.get("condition")
        if not self.condition:
            logger.warning(f"条件节点 {self.name}: 未提供条件表达式")
    
    def evaluate_condition(self, state: Dict[str, Any]) -> bool:
        """评估条件表达式"""
        if not self.condition:
            return True
        
        try:
            # 在状态上下文中评估条件
            context = dict(state)
            return eval(self.condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"条件节点 {self.name}: 评估条件 '{self.condition}' 失败: {e}")
            return False
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行节点逻辑，条件满足时处理"""
        try:
            if self.evaluate_condition(state):
                logger.info(f"条件节点 {self.name}: 条件满足，执行处理")
                return self.process(state)
            else:
                logger.info(f"条件节点 {self.name}: 条件不满足，跳过处理")
                return state
        except Exception as e:
            logger.error(f"条件节点 {self.name} 执行失败: {e}")
            state["error"] = str(e)
            return state