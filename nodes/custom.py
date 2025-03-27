from typing import Dict, Any, Callable
import inspect
from core.base import Node
from utils.logging import get_logger

logger = get_logger(__name__)


class FunctionNode(Node):
    """Function node, which wraps a regular function as a node."""
    
    def __init__(self, name: str, func: Callable, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.func = func
        self._validate_function()
    
    def _validate_function(self):
        """Validate whether the function has the correct signature."""
        sig = inspect.signature(self.func)
        if len(sig.parameters) != 1:
            logger.warning(f"Function node {self.name}: function should only accept one state parameter")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the function."""
        try:
            return self.func(state)
        except Exception as e:
            logger.error(f"Function node {self.name} execution failed: {e}")
            state["error"] = str(e)
            return state


class CustomNode(Node):
    """Custom node base class for easy user extension."""
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the state; users should override this method to implement custom logic.
        
        Args:
            state: Workflow state.
        
        Returns:
            The updated state.
        """
        raise NotImplementedError("Subclasses must implement the process method")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node logic."""
        try:
            return self.process(state)
        except Exception as e:
            logger.error(f"Custom node {self.name} execution failed: {e}")
            state["error"] = str(e)
            return state


class ConditionalNode(CustomNode):
    """Conditional node, which determines whether to execute processing based on a condition."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.condition = self.config.get("condition")
        if not self.condition:
            logger.warning(f"Conditional node {self.name}: condition expression not provided")
    
    def evaluate_condition(self, state: Dict[str, Any]) -> bool:
        """Evaluate the condition expression."""
        if not self.condition:
            return True
        
        try:
            # Evaluate the condition in the context of the state.
            context = dict(state)
            return eval(self.condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"Conditional node {self.name}: failed to evaluate condition '{self.condition}': {e}")
            return False
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node logic; process only if the condition is met."""
        try:
            if self.evaluate_condition(state):
                logger.info(f"Conditional node {self.name}: condition met, executing processing")
                return self.process(state)
            else:
                logger.info(f"Conditional node {self.name}: condition not met, skipping processing")
                return state
        except Exception as e:
            logger.error(f"Conditional node {self.name} execution failed: {e}")
            state["error"] = str(e)
            return state