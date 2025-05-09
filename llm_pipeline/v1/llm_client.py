"""LLM client for v1 pipeline."""

import json
import requests
import logging
import time
import traceback
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal, List
from llm_pipeline.config import LLM_API_URL, DEFAULT_MODEL_NAME, PROMPT_V1_PATH, MIXTRAL_API_URL
from llm_pipeline.common.io_utils import load_text, save_json
from llm_pipeline.common.llm_base import LLMClient
from llm_pipeline.common.log import setup_logger
from llm_pipeline.common.schema_utils import VALID_CATEGORIES
from llm_pipeline.common.metrics import APICallMetrics
from datetime import datetime
import string

# Set up logger
logger = setup_logger("llm_pipeline.v1.client", level=logging.WARNING)

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

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract JSON from a response that might be wrapped in a code block.
    
    Args:
        response_text: The raw response text from the API
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If no valid JSON is found in the response
    """
    # Try to find JSON in a code block first
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # If no code block or invalid JSON in code block, try parsing the whole response
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise ValueError("No valid JSON found in response")

def remove_hallucinated_numbers(numbers: List[str], source_text: str) -> tuple[List[str], List[str]]:
    """Filter out any phone numbers that do not appear verbatim in the source text.
    
    Args:
        numbers: List of phone numbers to check
        source_text: The original text to validate against
        
    Returns:
        Tuple of (valid_numbers, removed_numbers)
    """
    valid_numbers = []
    removed_numbers = []
    normalized_text = source_text.replace(" ", "")
    
    for num in numbers:
        if num.replace(" ", "") in normalized_text:
            valid_numbers.append(num)
        else:
            removed_numbers.append(num)
            logger.info(f"Removed hallucinated number: {num}")
    
    if removed_numbers:
        logger.info(f"Removed {len(removed_numbers)} hallucinated numbers: {removed_numbers}")
    
    return valid_numbers, removed_numbers

def parse_response_content(response: Dict[str, Any], source_text: str = "") -> List[str]:
    """
    Parse the LLM response to extract phone numbers.
    
    Args:
        response: The raw API response dictionary
        source_text: The original text that was processed (for validation)
        
    Returns:
        List of validated phone numbers
    """
    try:
        logger.debug(f"Parsing response: {json.dumps(response, indent=2)}")
        
        # Extract the content from the response
        if "choices" not in response:
            logger.debug("No 'choices' in response, attempting to use response directly")
            content = str(response)
        else:
            # Handle both Llama and Mixtral formats
            choice = response.get("choices", [{}])[0]
            if not choice:
                logger.debug("Empty choices array, attempting to use response directly")
                content = str(response)
            else:
                # Extract content based on format
                if "message" in choice:
                    # Mixtral chat format
                    content = choice.get("message", {}).get("content", "")
                else:
                    # Llama format
                    content = choice.get("text", "")
        
        if not content:
            logger.debug("Empty content, returning empty list")
            return []
            
        # Clean and normalize the content
        content = content.strip()
        logger.debug(f"Cleaned content: {content[:500]}...")
        
        # Check if no numbers were found
        if "NO_NUMBERS_FOUND" in content:
            logger.debug("LLM reported no numbers found")
            return []
            
        # Extract numbers from the response
        # Split into lines and look for lines that look like phone numbers
        lines = content.split('\n')
        found_numbers = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and non-number lines
            if not line or line.startswith('Step') or 'NUMBERS_FOUND' in line:
                continue
                
            # Check if the line looks like a phone number
            if re.search(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}', line):
                found_numbers.append(line)
                logger.debug(f"Found potential number: {line}")
        
        if not found_numbers:
            logger.debug("No phone numbers found in response")
            return []
            
        # Validate the numbers against source text
        validated_numbers = []
        for num in found_numbers:
            # Normalize both the number and source text for comparison
            # Remove all whitespace and special characters except digits and +
            normalized_num = re.sub(r'[^\d+]', '', num)
            normalized_source = re.sub(r'[^\d+]', '', source_text)
            
            if normalized_num in normalized_source:
                validated_numbers.append(num)  # Keep original formatting
                logger.debug(f"Validated number: {num}")
            else:
                logger.debug(f"Removed hallucinated number: {num}")
        
        if validated_numbers:
            logger.debug(f"Found {len(validated_numbers)} valid numbers: {validated_numbers}")
        else:
            logger.debug("No valid numbers found after validation")
            
        return validated_numbers
            
    except Exception as e:
        logger.error(f"Unexpected error in parse_response_content: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return []

# API configurations
API_CONFIGS = {
    "llama": {
        "url": LLM_API_URL,
        "payload_format": lambda prompt: {
            "model": "llama2",
            "prompt": prompt,
            "stream": False
        },
        "response_parser": parse_response_content
    },
    "mixtral": {
        "url": MIXTRAL_API_URL,
        "payload_format": lambda prompt: {
            "model": "mixtral-8x7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a phone number extraction assistant. You must respond with a valid JSON object containing phone numbers categorized as Sales, Support, Recruiting, General, or LowValue. The response must be a complete, valid JSON object with all categories present, even if empty."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
            "stop": ["</s>", "[INST]"],
            "format": "json"
        },
        "response_parser": parse_response_content
    }
}

def load_prompt() -> str:
    """Load the phone extraction prompt from file."""
    try:
        logger.debug(f"Loading prompt from {PROMPT_V1_PATH}")
        prompt_template = load_text(PROMPT_V1_PATH)
        logger.debug(f"Successfully loaded prompt template")
        logger.debug(f"Prompt template length: {len(prompt_template)} characters")
        logger.debug(f"First 500 chars of prompt template: {prompt_template[:500]}")
        
        # Check if the template already has the format string
        if '{webpage_text}' in prompt_template:
            logger.debug("Template already contains format string, using as is")
            return prompt_template
            
        # If not, try to replace the placeholder
        placeholder = '"""<INSERT YOUR TEXT HERE>"""'
        placeholder_pos = prompt_template.find(placeholder)
        logger.debug(f"Placeholder position: {placeholder_pos}")
        
        if placeholder_pos >= 0:
            logger.debug(f"Text around placeholder: {prompt_template[max(0, placeholder_pos-50):placeholder_pos+len(placeholder)+50]}")
            formatted_prompt = prompt_template.replace(placeholder, '{webpage_text}')
        else:
            logger.debug("Placeholder not found, adding format string at the end")
            formatted_prompt = prompt_template.rstrip() + '\n\n{webpage_text}\n\nRemember: Return ONLY the JSON object. No other text.'
            
        logger.debug(f"After formatting:")
        logger.debug(f"- Original length: {len(prompt_template)}")
        logger.debug(f"- New length: {len(formatted_prompt)}")
        logger.debug(f"- Format string present: {'{webpage_text}' in formatted_prompt}")
        
        # Ensure the prompt ends with the format string and the reminder
        expected_end = '{webpage_text}\n\nRemember: Return ONLY the JSON object. No other text.'
        if not formatted_prompt.endswith(expected_end):
            logger.debug(f"Prompt does not end with expected text")
            logger.debug(f"Current end: {formatted_prompt[-100:]}")
            logger.debug(f"Expected end: {expected_end}")
            formatted_prompt = formatted_prompt.rstrip() + '\n\n' + expected_end
            logger.debug(f"Added expected end text")
            
        logger.debug(f"Successfully formatted prompt template")
        logger.debug(f"Formatted prompt length: {len(formatted_prompt)} characters")
        logger.debug(f"First 500 chars of formatted prompt: {formatted_prompt[:500]}")
        logger.debug(f"Last 100 chars of formatted prompt: {formatted_prompt[-100:]}")
        
        return formatted_prompt
    except Exception as e:
        logger.error(f"Failed to load prompt: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        raise

def query_llm(prompt: str, api_type: API_TYPE = "llama", source_text: str = "") -> Dict[str, Any]:
    """Query the LLM API with the given prompt.
    
    Args:
        prompt: The prompt to send to the API
        api_type: The type of API to use ('llama' or 'mixtral')
        source_text: The original text being processed (for validation)
        
    Returns:
        The API response as a dictionary
        
    Raises:
        requests.RequestException: If there's an error making the API request
        json.JSONDecodeError: If the response is not valid JSON
    """
    if metrics_logger is None:
        logger.debug("Metrics logger not set - metrics will not be recorded")
    
    start_time = time.time()
    max_retries = 3
    base_delay = 1  # Base delay in seconds
    
    # Get API configuration
    api_config = API_CONFIGS.get(api_type)
    if not api_config:
        raise ValueError(f"Unsupported API type: {api_type}")
    
    # Log the first 300 characters of the prompt
    logger.debug(f"Sending prompt to {api_type} API (first 300 chars): {prompt[:300]}...")
    logger.debug(f"Full prompt length: {len(prompt)} characters")
    
    for attempt in range(max_retries):
        try:
            # Prepare the request payload
            if api_type == "mixtral":
                # Ollama format
                payload = {
                    "model": "mixtral",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False
                }
            else:
                # Original format for other APIs
                payload = {
                    "prompt": prompt,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "stop": ["</s>", "[INST]"]
                }
            
            # Make the API request
            logger.debug(f"Making request to {api_config['url']}")
            response = requests.post(
                api_config["url"],
                json=payload,
                timeout=120  # Increased timeout to 120 seconds
            )
            logger.debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            # Get raw response text for debugging
            raw_response_text = response.text
            logger.debug(f"Raw response text: {raw_response_text[:500]}...")
            
            # Parse response based on API type
            if api_type == "mixtral":
                try:
                    result = response.json()
                    # Extract content from Mixtral response
                    content = result.get("message", {}).get("content", "")
                    # Parse the content as JSON
                    try:
                        parsed_content = json.loads(content)
                        result = {"content": parsed_content}
                    except json.JSONDecodeError:
                        # If content is not JSON, use it as is
                        result = {"content": content}
                    
                    # Log metrics for Mixtral call
                    if metrics_logger:
                        metrics = APICallMetrics(
                            timestamp=datetime.now().isoformat(),
                            api_type=api_type,
                            prompt_length=len(prompt),
                            response_length=len(str(result)),
                            total_duration=time.time() - start_time
                        )
                        metrics_logger.log_api_call(metrics)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Mixtral response as JSON: {str(e)}")
                    logger.debug(f"Raw response text: {raw_response_text}")
                    raise
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
            content = api_config["response_parser"](result, source_text)
            logger.debug(f"Final parsed content: {content[:500]}...")
            
            return result
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            # If this is the last attempt, raise the exception
            if attempt == max_retries - 1:
                error_msg = f"Error querying {api_type} API after {max_retries} attempts: {str(e)}"
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
                    raise json.JSONDecodeError(f"Invalid JSON response from {api_type} API after {max_retries} attempts: {str(e)}", e.doc, e.pos)
            
            # Calculate delay with exponential backoff
            delay = base_delay * (2 ** attempt)
            logger.debug(f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
            time.sleep(delay)

def parse_llm_output(response: Dict[str, Any]) -> Union[List[str], str]:
    """
    Parse the LLM API response to extract the relevant content.
    
    Args:
        response: The raw API response dictionary
        
    Returns:
        Either a list of phone numbers or the raw text content
        
    Raises:
        ValueError: If the response format is invalid
    """
    return parse_response_content(response)

class V1LLMClient(LLMClient):
    """V1 LLM client that extends the base client with v1-specific behavior."""
    
    def __init__(self, prompt_path: Optional[Path] = None, api_type: API_TYPE = "llama"):
        """
        Initialize the v1 LLM client with default configuration.
        
        Args:
            prompt_path: Optional path to a custom prompt file. If not provided,
                        uses the default PROMPT_V1_PATH.
            api_type: The type of API to use ("llama" or "mixtral")
        """
        super().__init__(
            model_name=DEFAULT_MODEL_NAME,
            api_url=LLM_API_URL,
            prompt_path=prompt_path or PROMPT_V1_PATH,
            output_mode="simple"
        )
        self.api_type = api_type
    
    def generate(self, text: str) -> Union[List[str], str]:
        """
        Generate a response from the LLM for the given text.
        
        Args:
            text: The text to process
            
        Returns:
            Either a list of phone numbers or the raw text content
        """
        try:
            # Clean and format the input text
            cleaned_text = text.strip()
            if not cleaned_text:
                raise ValueError("Empty input text")
                
            logger.debug(f"Input text length: {len(cleaned_text)}")
            logger.debug(f"First 100 chars of input text: {cleaned_text[:100]}")
            
            # Format the prompt
            logger.debug(f"Prompt template before formatting: {self.prompt_template[:100]}...")
            logger.debug(f"Prompt template length: {len(self.prompt_template)}")
            logger.debug(f"Format string present in template: {'{webpage_text}' in self.prompt_template}")
            
            try:
                # Escape any curly braces in the text that aren't part of the format string
                escaped_text = cleaned_text.replace('{', '{{').replace('}', '}}')
                prompt = self.prompt_template.format(webpage_text=escaped_text)
                logger.debug(f"Successfully formatted prompt")
            except KeyError as e:
                logger.error(f"KeyError during prompt formatting: {str(e)}")
                logger.debug(f"Available format keys: {[k[1] for k in string.Formatter().parse(self.prompt_template) if k[1] is not None]}")
                raise
            except Exception as e:
                logger.error(f"Error during prompt formatting: {str(e)}")
                logger.debug("Full traceback:", exc_info=True)
                raise
            
            logger.debug(f"Formatted prompt for {self.api_type} (first 300 chars): {prompt[:300]}...")
            logger.debug(f"Full prompt length: {len(prompt)} characters")
            
            # Query the LLM
            response = query_llm(prompt, api_type=self.api_type, source_text=cleaned_text)
            logger.debug(f"Got response from LLM")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            
            # Parse the response
            result = self._parse_simple_output(response)
            logger.debug(f"Successfully parsed response")
            logger.debug(f"Parsed result: {json.dumps(result, indent=2)}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to build prompt: {str(e)}")
            logger.debug("Full traceback:", exc_info=True)
            raise

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
                
            # Ensure all required categories are present
            required_categories = ["Sales", "Support", "Recruiting", "General", "LowValue"]
            for category in required_categories:
                if category not in result:
                    raise ValueError(f"Missing required category: {category}")
                if not isinstance(result[category], list):
                    raise ValueError(f"Category '{category}' must contain a list of numbers")
                if not all(isinstance(num, str) for num in result[category]):
                    raise ValueError(f"Category '{category}' must contain only strings")
                    
            return result
            
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response content: {str(e)}")

def get_client() -> V1LLMClient:
    """
    Get a default-initialized V1LLMClient instance.
    
    Returns:
        A V1LLMClient instance configured with default settings.
        
    Example:
        >>> from llm_pipeline.v1.llm_client import get_client
        >>> client = get_client()
        >>> result = client.generate("Call us at (555) 123-4567")
    """
    return V1LLMClient()

def main() -> None:
    """Example usage of the v1 LLM client."""
    try:
        # Create client instance
        client = V1LLMClient()
        
        # Example text
        text = """
        Contact our sales team at +1 (800) 555-1234 for pricing information.
        For technical support, please call +1 (888) 555-5678.
        To apply for open positions, reach out to our HR department at +1 (877) 555-9012.
        Our main office number is (555) 123-4567.
        """
        
        # Generate results
        result = client.generate(text)
        
        # Log the result
        logger.info("LLM Response:")
        logger.debug(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.debug("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
