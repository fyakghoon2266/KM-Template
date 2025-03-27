from typing import Dict, Any, Callable
from langgraph.graph import StateGraph, END

from core.base import Node
from utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowBuilder:
    """Workflow builder for constructing LangGraph workflows"""
    
    def __init__(self, name: str = "default_workflow"):
        """
        Initialize workflow builder
        
        Args:
            name: Workflow name
        """
        self.name = name
        self.graph = StateGraph(state_schema=dict)
        self.nodes = {}
        self.has_entry_point = False
        self.leaf_nodes = set()
    
    def add_node(self, name: str, node: Node) -> Node:
        """
        Add node to workflow
        
        Args:
            name: Node name
            node: Node object
            
        Returns:
            Added node object
        """
        if name in self.nodes:
            logger.warning(f"Node {name} already exists, will be replaced")
        
        self.nodes[name] = node
        self.graph.add_node(name, node.to_runnable())
        logger.info(f"Added node: {name}")
        return node
    
    def set_entry_point(self, node_name: str):
        """
        Set entry point
        
        Args:
            node_name: Entry node name
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist, cannot set as entry point")
        
        self.graph.set_entry_point(node_name)
        self.has_entry_point = True
        logger.info(f"Set entry point: {node_name}")
    
    def add_edge(self, source: str, target: str):
        """
        Add edge
        
        Args:
            source: Source node name
            target: Target node name
        """
        if source not in self.nodes:
            raise ValueError(f"Source node {source} does not exist")
        if target not in self.nodes and target != END:
            raise ValueError(f"Target node {target} does not exist")
        
        self.graph.add_edge(source, target)
        logger.info(f"Added edge: {source} -> {target}")
    
    def add_conditional_edge(self, 
                            source: str, 
                            condition_fn: Callable[[Dict[str, Any]], bool], 
                            targets: Dict[bool, str]):
        """
        Add conditional edge
        
        Args:
            source: Source node name
            condition_fn: Condition function that receives state dict and returns boolean
            targets: Target nodes corresponding to condition results, format {True: node_name, False: node_name}
        """
        if source not in self.nodes:
            raise ValueError(f"Source node {source} does not exist")
        
        for result, target in targets.items():
            if target is not None and target != END and target not in self.nodes:
                raise ValueError(f"Target node {target} does not exist")
        
        self.graph.add_conditional_edges(source, condition_fn, targets)
        
        # Log condition info
        true_target = targets.get(True, "None")
        false_target = targets.get(False, "None")
        logger.info(f"Added conditional edge: {source} -> {true_target} (True) | {false_target} (False)")
    
    def mark_as_end(self, node_name: str):
        """
        Mark node as end point
        
        Args:
            node_name: Node name
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist")
        
        self.graph.add_edge(node_name, END)
        self.leaf_nodes.add(node_name)
        logger.info(f"Marked node {node_name} as end point")
    
    def compile(self) -> Callable:
        """
        編譯工作流
        
        Returns:
            編譯好的工作流
        """
        if not self.has_entry_point:
            raise ValueError("必須設置入口點才能編譯工作流")
        
        # 編譯工作流
        compiled_graph = self.graph.compile()
        logger.info(f"工作流 {self.name} 編譯成功")
        
        return compiled_graph
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run workflow
        
        Args:
            inputs: Input data
            
        Returns:
            Workflow execution result
        """
        compiled_graph = self.compile()
        logger.info(f"Running workflow {self.name}")
        return compiled_graph.invoke(inputs)