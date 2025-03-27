from typing import Dict, Any
from core.base import Node
from llm.langchain_models import LangChainLLM
from utils.logging import get_logger

logger = get_logger(__name__)

class LLMRouter(Node):
    """LLM router node, uses language model to make routing decisions"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Initialize LLM
        llm_config = self.config.get("llm", {})
        self.llm = LangChainLLM(llm_config)
        
        # Routing options
        self.route_options = self.config.get("route_options", [])
        if not self.route_options:
            raise ValueError(f"Node {self.name}: Must configure route options")
        
        # Get prompt information from config
        self.system_prompt = self.config.get("system_prompt", "You are a routing decision system that directs user queries to the most appropriate route.")
        self.prompt_template = self.config.get("prompt_template", "")
        
        # Default route
        self.default_route = self.config.get("default_route", self.route_options[0] if self.route_options else "default")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM routing logic"""
        query = state.get("query")
        if not query:
            logger.warning(f"Node {self.name}: No query text provided")
            state["route"] = self.default_route
            return state
        
        try:
            # Format routing options
            options_text = "\n".join([f"- {option}" for option in self.route_options])
            
            # Prepare prompt
            if self.prompt_template:
                prompt = self.prompt_template.format(
                    query=query,
                    route_options=options_text
                )
            else:
                prompt = f"Please select the most appropriate route from the following options:\n\n{options_text}\n\nUser query: {query}\n\nRespond with only the option name, without any explanation or additional text."
            
            # Get LLM decision
            response = self.llm.generate(prompt, system_message=self.system_prompt)
            
            # Extract route
            route = response.strip()
            
            # Verify route is in options
            if route in self.route_options:
                state["route"] = route
                logger.info(f"Node {self.name}: Routed to '{route}'")
            else:
                # Try fuzzy matching
                for option in self.route_options:
                    if option.lower() in route.lower():
                        state["route"] = option
                        logger.info(f"Node {self.name}: Fuzzy matched route to '{option}'")
                        return state
                
                # Use default route
                state["route"] = self.default_route
                logger.warning(f"Node {self.name}: Could not match route '{route}', using default route '{self.default_route}'")
            
            return state
            
        except Exception as e:
            logger.error(f"Node {self.name} routing failed: {e}")
            state["error"] = str(e)
            state["route"] = self.default_route
            return state


class ConditionalRouter(Node):
    """Conditional router node, determines next flow based on conditions"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Initialize conditions
        self.conditions = self.config.get("conditions", [])
        if not self.conditions:
            logger.warning(f"Node {self.name}: No routing conditions configured")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute routing logic"""
        # Default route result
        route_result = self.config.get("default_route", "default")
        
        # Evaluate conditions
        for condition in self.conditions:
            expression = condition.get("expression")
            result = condition.get("result")
            
            if not expression or not result:
                continue
            
            try:
                # Evaluate expression in current state context
                context = dict(state)
                if eval(expression, {"__builtins__": {}}, context):
                    route_result = result
                    logger.info(f"Node {self.name}: Condition '{expression}' met, routing to '{result}'")
                    break
            except Exception as e:
                logger.error(f"Node {self.name}: Error evaluating condition '{expression}': {e}")
        
        # Update route result
        state["route"] = route_result
        return state
