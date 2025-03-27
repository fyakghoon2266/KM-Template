import os
import yaml
from typing import Dict, Any
from pydantic import ValidationError

from core.types import WorkflowConfig
from utils.logging import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Configuration loader"""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file does not exist: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Process environment variables
            config = ConfigLoader._process_env_vars(config)
            return config
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
    
    @staticmethod
    def _process_env_vars(config: Any) -> Any:
        """Process environment variable references in configuration"""
        if isinstance(config, dict):
            return {k: ConfigLoader._process_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigLoader._process_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("\${") and config.endswith("}"):
            # Extract environment variable name
            env_var = config[2:-1]
            # Get environment variable value, return original string if not found
            return os.environ.get(env_var, config)
        else:
            return config
    
    @staticmethod
    def parse_workflow_config(config: Dict[str, Any]) -> WorkflowConfig:
        """Parse workflow configuration"""
        try:
            return WorkflowConfig(**config)
        except ValidationError as e:
            logger.error(f"Workflow configuration validation error: {e}")
            raise


class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self._config = None
    
    def load(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration"""
        path = config_path or self.config_path
        if not path:
            raise ValueError("Configuration file path not specified")
        
        self._config = ConfigLoader.load_yaml(path)
        return self._config
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration"""
        if self._config is None:
            raise ValueError("Configuration has not been loaded, call load() first")
        return self._config