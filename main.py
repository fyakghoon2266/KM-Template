import os
import argparse
from typing import Dict, Any, Optional
import importlib.util
import sys

from core.component_manager import ComponentManager
from utils.logging import get_logger, setup_file_logging
from utils.helpers import save_result

logger = get_logger(__name__)


def run_query(workflow, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute query"""
    # Prepare input state
    input_state = {"query": query}
    
    # Add additional parameters
    if parameters:
        input_state.update(parameters)
    
    # Execute workflow
    logger.info(f"Executing query: {query}")
    result = workflow.invoke(input_state)
    logger.info("Query execution completed")
    
    return result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG Framework Execution Tool")
    parser.add_argument("--config", "-c", required=True, help="Component configuration file path")
    parser.add_argument("--workflow", "-w", required=True, help="Workflow Python file path")
    parser.add_argument("--query", "-q", required=True, help="Query text")
    parser.add_argument("--output", "-o", default="./outputs", help="Output directory")
    parser.add_argument("--log", "-l", default="./logs", help="Log directory")
    
    args = parser.parse_args()
    
    # Setup file logging
    setup_file_logging(args.log)
    
    try:
        # Load component manager
        component_manager = ComponentManager(args.config)
        
        # Dynamically load workflow
        spec = importlib.util.spec_from_file_location("workflow_module", args.workflow)
        workflow_module = importlib.util.module_from_spec(spec)
        sys.modules["workflow_module"] = workflow_module
        spec.loader.exec_module(workflow_module)
        
        # Check if module has build_workflow function
        if not hasattr(workflow_module, "build_workflow"):
            raise ValueError(f"Workflow file {args.workflow} does not define build_workflow function")
        
        # Build workflow
        workflow = workflow_module.build_workflow(component_manager)
        
        # Execute query
        result = run_query(workflow, args.query)
        
        # Save result
        save_result(result, args.output)

        print(result)
        
        # Output result
        if "answer" in result:
            print("\nAnswer:\n", result["answer"])
        else:
            print("\nResult:\n", result)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()