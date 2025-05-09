"""Base LLM client class for phone number extraction."""

import json
import logging
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal, List
from llm_pipeline.common.schema_utils import validate_output
from llm_pipeline.common.io_utils import load_text
from llm_pipeline.common.log import setup_logger

class LLMClient:
    """Base class for LLM clients that handles both v1 and v2 output styles."""
    
    def __init__(
        self,
        model_name: str,
        api_url: str,
        prompt_path: Path,
        output_mode: Literal["simple", "structured"] = "structured",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        request_timeout: int = 20,  # Default timeout of 20 seconds
        system_prompt: Optional[str] = None  # Optional custom system prompt
    ):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the LLM model to use
            api_url: URL of the LLM API endpoint
            prompt_path: Path to the prompt template file
            output_mode: Either "simple" (v1) or "structured" (v2) output format
            temperature: Sampling temperature for the LLM (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            request_timeout: Timeout in seconds for API requests (default: 20)
            system_prompt: Optional custom system prompt to use instead of the default
        """
        self.model_name = model_name
        self.api_url = api_url
        self.prompt_path = prompt_path
        self.output_mode = output_mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.system_prompt = system_prompt or "You are a helpful assistant that extracts phone numbers from text."
        
        # Set up logger
        self.logger = setup_logger("llm_pipeline.client", level=logging.DEBUG)
        
        # Load the prompt template
        try:
            prompt_template = load_text(prompt_path)
            # Replace the placeholder with a format string
            self.prompt_template = prompt_template.replace('"""<INSERT YOUR TEXT HERE>"""', '{webpage_text}')
            self.logger.debug(f"Successfully loaded and formatted prompt template")
            self.logger.debug(f"Prompt template length: {len(self.prompt_template)} characters")
            self.logger.debug(f"First 500 chars of prompt template: {self.prompt_template[:500]}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
        except Exception as e:
            raise Exception(f"Error loading prompt: {str(e)}")
    
    def __repr__(self) -> str:
        """
        Return a string representation of the LLM client configuration.
        
        Returns:
            A string showing the model name, API URL, and output mode
        """
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"api_url='{self.api_url}', "
            f"output_mode='{self.output_mode}'"
            f")"
        )
    
    def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Send a POST request to the LLM API with the given prompt.
        Implements exponential backoff retry logic for failed requests.
        
        Args:
            prompt: The prompt text to send to the LLM
            
        Returns:
            Dict containing the raw API response
            
        Raises:
            requests.RequestException: If the HTTP request fails after all retries
        """
        # OpenAI-style request format
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Log the prompt being sent (truncated to 500 chars)
        self.logger.debug(f"Sending prompt to LLM (truncated): {prompt[:500]}...")
        
        max_retries = 3
        base_delay = 2  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=self.request_timeout)
                response.raise_for_status()
                result = response.json()
                
                # Log the raw response (truncated to 500 chars)
                response_str = json.dumps(result)
                self.logger.debug(f"Received LLM response (truncated): {response_str[:500]}...")
                
                return result
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    if isinstance(e, requests.RequestException):
                        raise requests.RequestException(f"Failed to query LLM API after {max_retries} attempts: {str(e)}")
                    else:
                        raise json.JSONDecodeError(f"Invalid JSON response from API after {max_retries} attempts: {str(e)}", e.doc, e.pos)
                
                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)
                self.logger.debug(f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                time.sleep(delay)
    
    def _parse_simple_output(self, response: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Parse the LLM response in v1 (simple) format.
        
        Args:
            response: The raw API response dictionary
            
        Returns:
            Dict with categories as keys and lists of phone numbers as values
            
        Raises:
            ValueError: If the response format is invalid
        """
        try:
            # Extract the content from the response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty response content")
                
            # Parse the JSON content
            result = json.loads(content)
            
            # Validate the structure
            if not isinstance(result, dict):
                raise ValueError("Response must be a dictionary")
                
            # Ensure all values are lists
            for category, numbers in result.items():
                if not isinstance(numbers, list):
                    raise ValueError(f"Category '{category}' must contain a list of numbers")
                if not all(isinstance(num, str) for num in numbers):
                    raise ValueError(f"Category '{category}' must contain only strings")
                    
            return result
            
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response content: {str(e)}")
    
    def _parse_structured_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM response in v2 (structured) format.
        
        Args:
            response: The raw API response dictionary
            
        Returns:
            Dict containing:
                - phone_numbers: List of phone number objects with number, category, confidence, and context
                - metadata: Dict with total_numbers_found and processing_timestamp
                
        Raises:
            ValueError: If the response format is invalid or missing required keys
        """
        try:
            # Extract the content from the response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty response content")
                
            # Parse the JSON content
            result = json.loads(content)
            
            # Validate top-level structure
            if not isinstance(result, dict):
                raise ValueError("Response must be a dictionary")
            
            # Check for required top-level keys
            if "phone_numbers" not in result:
                raise ValueError("Response missing required 'phone_numbers' key")
            if "metadata" not in result:
                raise ValueError("Response missing required 'metadata' key")
            
            # Validate types of top-level keys
            if not isinstance(result["phone_numbers"], list):
                raise ValueError("'phone_numbers' must be a list")
            if not isinstance(result["metadata"], dict):
                raise ValueError("'metadata' must be a dictionary")
            
            # Add timestamp if not present
            if "processing_timestamp" not in result["metadata"]:
                result["metadata"]["processing_timestamp"] = datetime.utcnow().isoformat()
                
            # Add total count if not present
            if "total_numbers_found" not in result["metadata"]:
                result["metadata"]["total_numbers_found"] = len(result["phone_numbers"])
                
            # Validate the output against the schema
            validate_output(result)
                
            return result
                
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response content: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Schema validation failed: {str(e)}")
    
    def generate(self, input_text: str, dry_run: bool = False) -> Union[Dict[str, List[str]], Dict[str, Any], str]:
        """
        Generate phone number extraction results from input text.
        
        Args:
            input_text: The text to extract phone numbers from
            dry_run: If True, return the constructed prompt instead of making API call
            
        Returns:
            If dry_run is False:
                Either a simple dict (v1) or structured dict (v2) containing the extracted phone numbers
            If dry_run is True:
                The constructed prompt string
            
        Raises:
            ValueError: If the response format is invalid
            requests.RequestException: If the API request fails
        """
        # Build the prompt by formatting the template
        prompt = self.prompt_template.format(webpage_text=input_text)
        
        # If dry run, return the prompt
        if dry_run:
            self.logger.debug(f"Dry run - returning constructed prompt: {prompt[:500]}...")
            return prompt
        
        # Query the LLM
        response = self._query_llm(prompt)
        
        # Parse the output based on the selected mode
        if self.output_mode == "simple":
            return self._parse_simple_output(response)
        else:  # structured
            return self._parse_structured_output(response) 