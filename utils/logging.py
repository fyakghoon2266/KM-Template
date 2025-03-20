import logging
import sys
import os
from typing import Optional
from pathlib import Path


def get_logger(name: str, level: int = None) -> logging.Logger:
    """
    Get a configured logger
    
    Args:
        name: Logger name
        level: Log level, defaults to INFO
    
    Returns:
        Configured logger
    """
    if level is None:
        # Get log level from environment variable, default to INFO
        level_name = os.environ.get("LOG_LEVEL", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate configuration
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger


def setup_file_logging(log_dir: Optional[str] = None, name: Optional[str] = None) -> None:
    """
    Setup file logging
    
    Args:
        log_dir: Log directory, defaults to logs in current directory
        name: Log file name prefix, defaults to rag_framework
    """
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    
    # Ensure log directory exists
    log_dir.mkdir(exist_ok=True, parents=True)
    
    name = name or "rag_framework"
    log_file = log_dir / f"{name}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)