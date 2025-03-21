from typing import Dict, Any, List, Optional, Union
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()
from utils.logging import get_logger

logger = get_logger(__name__)


class LangChainEmbeddings:
    """LangChain embeddings model wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding model
        
        Args:
            config: Configuration dictionary with model parameters
                - model: Model name, defaults to "text-embedding-ada-002"
                - api_key: OpenAI API key (optional)
                - api_base: OpenAI API base URL (optional)
                - other_params: Other parameters passed directly to OpenAIEmbeddings
        """

        # Extract configuration
        kwargs = {k: v for k, v in config.items() 
                if k not in ["model", "api_key", "api_base"]}
        
        # Initialize embedding model
   
        try:
            self.embedding_model = OpenAIEmbeddings(
                model=os.environ['OPENAI_EMBEDDING_DEPLOYMENT'],
                openai_api_key=os.environ['OPENAI_API_KEY'],
                base_url=os.environ['OPENAI_API_BASE'],
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text"""
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed document texts"""
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            raise


class LangChainLLM:
    """LangChain LLM model wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM model
        
        Args:
            config: Configuration dictionary with model parameters
                - model: Model name, defaults to "gpt-3.5-turbo"
                - temperature: Temperature parameter
                - max_tokens: Maximum tokens to generate
                - api_key: OpenAI API key (optional)
                - api_base: OpenAI API base URL (optional)
                - other_params: Other parameters passed directly to ChatOpenAI
        """
        # Extract configuration

        kwargs = {k: v for k, v in config.items() 
                if k not in ["model", "api_key", "api_base"]}
        
        # Initialize LangChain LLM
        try:
            self.llm = ChatOpenAI(
                openai_api_base=os.environ['OPENAI_API_BASE'],
                api_key=os.environ['OPENAI_API_KEY'],
                model=os.environ['OPENAI_DEPLOYMENT'],
                **kwargs
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            # Provide default fallback
            self.llm = None
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """
        Generate text
        
        Args:
            prompt: User prompt
            system_message: System message (optional)
            kwargs: Additional parameters passed to LLM
        
        Returns:
            Generated text response
        """
        try:
            # Ensure LLM is initialized
            if self.llm is None:
                return "LLM is not available. Please check API configuration."
                
            # Prepare messages
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            # Call LLM
            response = self.llm.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return a default response instead of raising an exception
            return f"Unable to generate a response: An error occurred connecting to the AI service. Please check API credentials and network connection."
    
    def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate text with message list
        
        Args:
            messages: Message list in format [{"role": "system"|"user"|"assistant", "content": "message content"}]
            kwargs: Additional parameters passed to LLM
            
        Returns:
            Generated text response
        """
        try:
            # Ensure LLM is initialized
            if self.llm is None:
                return "LLM is not available. Please check API configuration."
                
            # Convert message format
            langchain_messages = []
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                # assistant messages typically used as history, LangChain handles them
            
            # Call LLM
            response = self.llm.invoke(langchain_messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return a default response
            return f"Unable to generate a response: An error occurred connecting to the AI service."