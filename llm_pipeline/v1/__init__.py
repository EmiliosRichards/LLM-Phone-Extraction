"""
LLM Pipeline v1 - Basic phone number extraction.
"""

from typing import List

# Load the v1 prompt template from disk
from llm_pipeline.v1.llm_client import load_prompt
# Send a query to the LLM and get the response
from llm_pipeline.v1.llm_client import query_llm
# Parse the LLM's output into structured data
from llm_pipeline.v1.llm_client import parse_llm_output
# Get a configured LLM client instance
from llm_pipeline.v1.llm_client import V1LLMClient

def get_client() -> V1LLMClient:
    """Get a configured LLM client instance ready for use."""
    return V1LLMClient()

__all__: List[str] = [
    'get_client',
    'load_prompt',
    'parse_llm_output',
    'query_llm'
] 