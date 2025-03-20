from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field


class Node(ABC):
    """Base class for workflow nodes"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node logic"""
        pass
    
    def to_runnable(self) -> Callable:
        """Convert node to LangGraph runnable object"""
        def runnable(state):
            return self(state)
        return runnable


class Edge(BaseModel):
    """Workflow edge definition"""
    source: str
    target: str
    condition: Optional[str] = None


class WorkflowSchema(BaseModel):
    """Workflow schema definition"""
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]]
    edges: List[Edge]
    entry_point: str
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class VectorStore(ABC):
    """Base class for vector stores"""
    
    @abstractmethod
    def query(self, query_vector: List[float], collection_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query for most similar documents"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], collection_name: str) -> List[str]:
        """Add documents to vector store"""
        pass