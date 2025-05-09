"""LLM client for v2 pipeline with structured output."""

import json
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from llm_pipeline.config import LLM_API_URL, DEFAULT_MODEL_NAME, PROMPT_V2_PATH
from llm_pipeline.common.schema_utils import validate_output
from llm_pipeline.common.llm_base import LLMClient
from llm_pipeline.common.log import setup_logger
from llm_pipeline.common.metrics import APICallMetrics
import logging
import time
import re

# Set up logger
logger = setup_logger("llm_pipeline.v2", level=logging.DEBUG)

# API Types
API_TYPE = Literal["llama", "mixtral"]

# Global metrics logger instance - will be set by main module
metrics_logger = None

def set_metrics_logger(logger_instance):
    """Set the global metrics logger instance.
    
    Args:
        logger_instance: The MetricsLogger instance to use
    """
    global metrics_logger
    metrics_logger = logger_instance

# API Configuration
API_CONFIGS = {
    "llama": {
        "url": LLM_API_URL,
        "model": DEFAULT_MODEL_NAME,
        "payload_format": lambda prompt: {
            "model": DEFAULT_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts phone numbers from text."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "response_parser": lambda response: response.get("choices", [{}])[0].get("message", {}).get("content", "")
    },
    "mixtral": {
        "url": "http://localhost:22760/api/generate",
        "model": "mixtral",
        "payload_format": lambda prompt: {
            "model": "mixtral",
            "prompt": f"""You are a phone number extraction assistant. Your task is to extract phone numbers from text and return them in a structured JSON format.

IMPORTANT: You must ALWAYS return a valid JSON object, even if no phone numbers are found. Do not include any other text or explanation.

Instructions:
1. Identify and extract all phone numbers
2. Categorize each number into one of these categories:
   - "Sales": Numbers related to purchasing, business inquiries, or account managers
   - "Support": Customer service or technical help
   - "Recruiting": HR, careers, or hiring lines
   - "General": Company-wide numbers or reception
   - "LowValue": Spam, ads, fake, incomplete or clearly irrelevant numbers

3. For each phone number, provide:
   - The number in E.164 format when possible
   - The category it belongs to
   - A confidence score between 0 and 1
   - The surrounding context (up to 100 characters)

4. Include metadata with:
   - Total count of unique phone numbers found
   - ISO 8601 timestamp of when the extraction was performed

The response must be a valid JSON object with this exact structure:
{{
  "phone_numbers": [
    {{
      "number": string,  // The phone number in E.164 format
      "category": string,  // One of: "Sales", "Support", "Recruiting", "General", "LowValue"
      "confidence": number,  // Between 0 and 1
      "context": string  // Up to 100 characters of surrounding text
    }}
  ],
  "metadata": {{
    "total_numbers_found": number,
    "processing_timestamp": string  // ISO 8601 format
  }}
}}

Example 1 - No phone numbers found:
{{
  "phone_numbers": [],
  "metadata": {{
    "total_numbers_found": 0,
    "processing_timestamp": "2024-03-20T15:30:00Z"
  }}
}}

Example 2 - Single phone number:
{{
  "phone_numbers": [
    {{
      "number": "+1 (800) 555-1234",
      "category": "Sales",
      "confidence": 0.95,
      "context": "Contact our sales team at +1 (800) 555-1234 for pricing information"
    }}
  ],
  "metadata": {{
    "total_numbers_found": 1,
    "processing_timestamp": "2024-03-20T15:30:00Z"
  }}
}}

Example 3 - Multiple phone numbers:
{{
  "phone_numbers": [
    {{
      "number": "+1 (800) 555-1234",
      "category": "Sales",
      "confidence": 0.95,
      "context": "Contact our sales team at +1 (800) 555-1234 for pricing information"
    }},
    {{
      "number": "+1 (888) 555-5678",
      "category": "Support",
      "confidence": 0.9,
      "context": "For technical support, please call +1 (888) 555-5678"
    }}
  ],
  "metadata": {{
    "total_numbers_found": 2,
    "processing_timestamp": "2024-03-20T15:30:00Z"
  }}
}}

Text to process:

{prompt}

Remember: You must return ONLY a valid JSON object. No other text or explanation.""",
            "stream": False,
            "temperature": 0.1,  # Lower temperature for more consistent JSON output
            "max_tokens": 1000
        },
        "response_parser": lambda response: response.get("response", "")
    }
}

def load_prompt() -> str:
    """Load the structured phone extraction prompt from file."""
    try:
        with open(PROMPT_V2_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at {PROMPT_V2_PATH}")
    except Exception as e:
        raise Exception(f"Error loading prompt: {str(e)}")

def parse_structured_output(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the LLM response in v2 format with schema validation.
    
    Args:
        response: The raw API response dictionary
        
    Returns:
        Dict containing:
            - phone_numbers: List of phone number objects with number, category, confidence, and context
            - metadata: Dict with total_numbers_found and processing_timestamp
            
    Raises:
        ValueError: If the response format is invalid or schema validation fails
    """
    try:
        # Extract the content from the response
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            logger.error("Empty response content received from LLM")
            raise ValueError("Empty response content")
            
        # Parse the JSON content
        logger.debug("Parsing JSON content from LLM response")
        result = json.loads(content)
        
        # Add timestamp if not present
        if "processing_timestamp" not in result["metadata"]:
            logger.debug("Adding missing processing timestamp")
            result["metadata"]["processing_timestamp"] = datetime.utcnow().isoformat()
            
        # Add total count if not present
        if "total_numbers_found" not in result["metadata"]:
            logger.debug("Adding missing total_numbers_found count")
            result["metadata"]["total_numbers_found"] = len(result["phone_numbers"])
        
        # Convert confidence values to float if they aren't already
        for phone in result["phone_numbers"]:
            if "confidence" in phone:
                phone["confidence"] = float(phone["confidence"])
        
        # Sort metadata keys for consistent formatting
        result["metadata"] = dict(sorted(result["metadata"].items()))
            
        # Validate the output against the schema
        logger.debug("Validating output against schema")
        try:
            validate_output(result)
        except ValueError as e:
            # Include the malformed result in the error message
            error_msg = f"Schema validation failed: {str(e)}\nMalformed result: {json.dumps(result, indent=2)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log success with number of phone numbers found
        logger.info(f"Successfully parsed and validated response with {result['metadata']['total_numbers_found']} phone numbers")
        return result
            
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid response format: {str(e)}")
        raise ValueError(f"Invalid response format: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in response content: {str(e)}")
        raise ValueError(f"Invalid JSON in response content: {str(e)}")

def query_llm_v2(prompt: str, api_type: API_TYPE = "llama", retry: int = 3, timeout: int = 20) -> Dict[str, Any]:
    """
    Send a POST request to the LLM API with the given prompt.
    
    Args:
        prompt: The prompt text to send to the LLM
        api_type: The type of API to use ("llama" or "mixtral")
        retry: Number of retry attempts (default: 3)
        timeout: Request timeout in seconds (default: 20)
        
    Returns:
        Dict containing the raw API response
        
    Raises:
        requests.RequestException: If the HTTP request fails after all retries
        json.JSONDecodeError: If the response is not valid JSON after all retries
    """
    if metrics_logger is None:
        logger.warning("Metrics logger not set - metrics will not be recorded")
    
    start_time = time.time()
    
    # Get API configuration
    api_config = API_CONFIGS.get(api_type)
    if not api_config:
        raise ValueError(f"Unsupported API type: {api_type}")
    
    # Log the first 300 characters of the prompt
    logger.debug(f"Sending prompt to {api_type} API (first 300 chars): {prompt[:300]}...")
    
    # Format payload according to API type
    payload = api_config["payload_format"](prompt)
    
    base_delay = 1
    
    for attempt in range(retry):
        try:
            response = requests.post(api_config["url"], json=payload, timeout=timeout)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
            
            # For Mixtral, we need to handle the response differently
            if api_type == "mixtral":
                # Get the raw text response
                content = response.text.strip()
                logger.debug(f"Raw Mixtral response: {content[:500]}...")
                
                try:
                    # First try to parse the entire response as JSON
                    response_json = json.loads(content)
                    # Extract the actual response content
                    content = response_json.get("response", "").strip()
                    logger.debug(f"Extracted content from Mixtral response: {content[:500]}...")
                    
                    # Try to extract JSON from the response content
                    try:
                        # First try to find JSON in code blocks
                        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
                        if json_match:
                            json_str = json_match.group(1)
                            logger.debug(f"Found JSON in code block: {json_str[:500]}...")
                            json_content = json.loads(json_str)
                        else:
                            # Try parsing the whole content as JSON
                            logger.debug("Trying to parse entire content as JSON...")
                            json_content = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON parsing failed: {str(e)}")
                        # If JSON parsing fails, try to extract just the JSON object
                        json_match = re.search(r'(\{[\s\S]*\})', content)
                        if json_match:
                            try:
                                json_str = json_match.group(1)
                                logger.debug(f"Found JSON object: {json_str[:500]}...")
                                json_content = json.loads(json_str)
                            except json.JSONDecodeError:
                                # If still failing, try to clean the JSON string
                                cleaned_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                                logger.debug(f"Cleaned JSON: {cleaned_json[:500]}...")
                                json_content = json.loads(cleaned_json)
                        else:
                            # If no JSON found, return empty result
                            logger.warning("No JSON found in response, returning empty result")
                            json_content = {
                                "phone_numbers": [],
                                "metadata": {
                                    "total_numbers_found": 0,
                                    "processing_timestamp": datetime.now().isoformat()
                                }
                            }
                    
                    # Create a response object that matches our expected format
                    result = {
                        "choices": [{
                            "message": {
                                "content": json.dumps(json_content, ensure_ascii=False)
                            }
                        }]
                    }
                    
                    # Log metrics for successful Mixtral call
                    if metrics_logger:
                        metrics = APICallMetrics(
                            timestamp=datetime.now().isoformat(),
                            api_type=api_type,
                            prompt_length=len(prompt),
                            response_length=len(content),
                            total_duration=time.time() - start_time,
                            load_duration=response_json.get("load_duration"),
                            prompt_eval_count=response_json.get("prompt_eval_count"),
                            prompt_eval_duration=response_json.get("prompt_eval_duration"),
                            eval_count=response_json.get("eval_count"),
                            eval_duration=response_json.get("eval_duration")
                        )
                        metrics_logger.log_api_call(metrics)
                    
                except (ValueError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to extract JSON from Mixtral response: {str(e)}")
                    # Return empty result instead of raising error
                    result = {
                        "choices": [{
                            "message": {
                                "content": json.dumps({
                                    "phone_numbers": [],
                                    "metadata": {
                                        "total_numbers_found": 0,
                                        "processing_timestamp": datetime.now().isoformat(),
                                        "error": str(e)
                                    }
                                }, ensure_ascii=False)
                            }
                        }]
                    }
                    
                    # Log metrics for failed Mixtral call
                    if metrics_logger:
                        metrics = APICallMetrics(
                            timestamp=datetime.now().isoformat(),
                            api_type=api_type,
                            prompt_length=len(prompt),
                            response_length=0,
                            total_duration=time.time() - start_time,
                            success=False,
                            error_message=str(e)
                        )
                        metrics_logger.log_api_call(metrics)
            else:
                result = response.json()
                # Log metrics for Llama call
                if metrics_logger:
                    metrics = APICallMetrics(
                        timestamp=datetime.now().isoformat(),
                        api_type=api_type,
                        prompt_length=len(prompt),
                        response_length=len(str(result)),
                        total_duration=time.time() - start_time
                    )
                    metrics_logger.log_api_call(metrics)
            
            # Extract content using API-specific parser
            content = api_config["response_parser"](result)
            logger.debug(f"Received {api_type} response (trimmed): {content[:300]}...")
            
            return result
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            # If this is the last attempt, raise the exception
            if attempt == retry - 1:
                error_msg = f"Error querying {api_type} API after {retry} attempts: {str(e)}"
                logger.error(error_msg)
                logger.debug("Full traceback:", exc_info=True)
                
                # Log metrics for failed call
                if metrics_logger:
                    metrics = APICallMetrics(
                        timestamp=datetime.now().isoformat(),
                        api_type=api_type,
                        prompt_length=len(prompt),
                        response_length=0,
                        total_duration=time.time() - start_time,
                        success=False,
                        error_message=error_msg
                    )
                    metrics_logger.log_api_call(metrics)
                
                if isinstance(e, requests.RequestException):
                    raise requests.RequestException(error_msg)
                else:
                    raise json.JSONDecodeError(f"Invalid JSON response from {api_type} API after {retry} attempts: {str(e)}", e.doc, e.pos)
            
            # Calculate delay with exponential backoff
            delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s, etc.
            logger.warning(f"Request failed (attempt {attempt + 1}/{retry}). Retrying in {delay} seconds...")
            time.sleep(delay)

class V2LLMClient(LLMClient):
    """V2 LLM client that extends the base client with structured output and schema validation."""
    
    def __init__(self, prompt_path: Optional[Path] = None, api_type: API_TYPE = "llama"):
        """
        Initialize the v2 LLM client with default configuration.
        
        Args:
            prompt_path: Optional custom path to prompt template file.
                        If not provided, uses the default PROMPT_V2_PATH.
            api_type: The type of API to use ("llama" or "mixtral")
        """
        super().__init__(
            model_name=DEFAULT_MODEL_NAME,
            api_url=LLM_API_URL,
            prompt_path=prompt_path or PROMPT_V2_PATH,
            output_mode="structured"
        )
        self.api_type = api_type
    
    def generate(self, text: str) -> Dict[str, Any]:
        """
        Generate a response from the LLM for the given text.
        
        Args:
            text: The text to process
            
        Returns:
            Dict containing the structured output
        """
        prompt = self.prompt_template.format(webpage_text=text)
        response = query_llm_v2(prompt, api_type=self.api_type)
        return parse_structured_output(response)

def get_client_v2(prompt_path: Optional[Path] = None, api_type: API_TYPE = "llama") -> V2LLMClient:
    """
    Get a default-initialized V2LLMClient instance.
    
    Args:
        prompt_path: Optional custom path to prompt template file.
                    If not provided, uses the default PROMPT_V2_PATH.
        api_type: The type of API to use ("llama" or "mixtral")
                    
    Returns:
        A configured V2LLMClient instance
    """
    return V2LLMClient(prompt_path=prompt_path, api_type=api_type)

def main():
    """Example usage of the v2 LLM client."""
    try:
        # Create client instance
        client = V2LLMClient()
        
        # Example text
        text = """
        Contact our sales team at +1 (800) 555-1234 for pricing information.
        For technical support, please call +1 (888) 555-5678.
        To apply for open positions, reach out to our HR department at +1 (877) 555-9012.
        Our main office number is (555) 123-4567.
        """
        
        # Measure generation time
        start_time = time.monotonic()
        result = client.generate(text)
        latency = time.monotonic() - start_time
        
        # Log the latency
        logger.info(f"LLM generation completed in {latency:.2f} seconds")
        
        # Print the result
        print("LLM Response:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 