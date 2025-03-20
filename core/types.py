from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class NodeConfig(BaseModel):
    """Node configuration"""
    name: str
    type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EdgeConfig(BaseModel):
    """Edge configuration"""
    source: str
    target: str
    condition: Optional[str] = None


class RAGConfig(BaseModel):
    """RAG configuration"""
    collections: List[str]
    embedder: Dict[str, Any]
    top_k: int = 3


class WorkflowConfig(BaseModel):
    """Workflow configuration"""
    name: str
    description: Optional[str] = None
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]
    entry_point: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Document model"""
    id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Query result"""
    query: str
    documents: List[Document]
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)