import os
import json
from typing import Dict, Any, List, Optional
import yaml
import time
from pathlib import Path

from core.types import Document
from utils.logging import get_logger

logger = get_logger(__name__)


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load YAML config: {e}")
        raise


def save_yaml_config(config: Dict[str, Any], file_path: str) -> None:
    """Save YAML configuration file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    except Exception as e:
        logger.error(f"Failed to save YAML config: {e}")
        raise


def load_documents_from_json(file_path: str) -> List[Document]:
    """Load documents from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                id=item.get("id"),
                content=item.get("content", ""),
                metadata=item.get("metadata", {})
            )
            documents.append(doc)
        
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise


def save_result(result: Dict[str, Any], output_dir: str, filename: Optional[str] = None) -> str:
    """
    Save query result
    
    Args:
        result: Query result
        output_dir: Output directory
        filename: Filename, defaults to timestamp
    
    Returns:
        Saved file path
    """
    # Ensure directory exists
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Default to timestamp as filename
    if filename is None:
        timestamp = int(time.time())
        filename = f"result_{timestamp}.json"
    
    file_path = os.path.join(output_dir, filename)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Result saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save result: {e}")
        raise


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for log output"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."