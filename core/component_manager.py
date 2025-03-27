from typing import Dict, Any
import os

from core.config import ConfigLoader
from utils.logging import get_logger

logger = get_logger(__name__)


class ComponentManager:
    """Component manager that loads component definitions from YAML"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.components = {}  # Store all predefined components
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load component configuration"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
        
        return ConfigLoader.load_yaml(config_path)
    
    def _initialize_components(self):
        """Initialize all components"""
        # Preload shared components if needed
        pass
    
    def create_retriever(self, name: str, **kwargs):
        """Create a retriever instance"""
        from nodes.retriever import QdrantRetriever
        
        # Get base configuration
        base_config = {}
        retrievers_config = self.config.get("retrievers", {})
        
        if name in retrievers_config:
            base_config = retrievers_config[name]
        elif "default" in retrievers_config:
            base_config = retrievers_config["default"]
        
        # Merge runtime parameters
        config = {**base_config, **kwargs}
        
        # Create instance
        return QdrantRetriever(name, config)
    
    def create_generator(self, name: str, **kwargs):
        """Create a generator instance"""
        from nodes.generator import LLMGenerator
        
        # Get base configuration
        base_config = {}
        generators_config = self.config.get("generators", {})
        
        if name in generators_config:
            base_config = generators_config[name]
        elif "default" in generators_config:
            base_config = generators_config["default"]
        
        # Merge runtime parameters
        config = {**base_config, **kwargs}
        
        # Create instance
        return LLMGenerator(name, config)
    
    def create_router(self, name: str, **kwargs):
        """Create a router instance"""
        from nodes.router import LLMRouter
        
        # Get base configuration
        base_config = {}
        routers_config = self.config.get("routers", {})
        
        if name in routers_config:
            base_config = routers_config[name]
        elif "default" in routers_config:
            base_config = routers_config["default"]
        
        # Merge runtime parameters
        config = {**base_config, **kwargs}
        
        # Create instance
        return LLMRouter(name, config)
    
    def create_reranker(self, name: str, **kwargs):
        """Create a reranker instance"""
        from nodes.reranker import SimpleReranker, LLMReranker
        
        # Get base configuration
        base_config = {}
        rerankers_config = self.config.get("rerankers", {})
        
        if name in rerankers_config:
            base_config = rerankers_config[name]
        elif "default" in rerankers_config:
            base_config = rerankers_config["default"]
        
        # Merge runtime parameters
        config = {**base_config, **kwargs}
        
        # Determine reranker type
        reranker_type = config.get("type", "simple")
        
        # Create appropriate reranker
        if reranker_type.lower() == "llm":
            return LLMReranker(name, config)
        else:
            return SimpleReranker(name, config)
    
    def create_custom_node(self, node_type: str, name: str, **kwargs):
        """Create a custom node instance"""
        # Import dynamically based on node type
        parts = node_type.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        
        try:
            module = __import__(module_path, fromlist=[class_name])
            node_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Cannot import node type {node_type}: {e}")
            raise ValueError(f"Unknown node type: {node_type}")
        
        # Get base configuration
        base_config = {}
        custom_config = self.config.get("custom_nodes", {}).get(name, {})
        
        # Merge runtime parameters
        config = {**base_config, **custom_config, **kwargs}
        
        # Create instance
        return node_class(name, config)