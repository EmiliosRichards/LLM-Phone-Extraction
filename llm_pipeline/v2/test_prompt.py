"""Test the LLM prompt with example text."""

import argparse
import json
import traceback
from pathlib import Path
from llm_pipeline.v2.llm_client import query_llm_v2, parse_structured_output, API_TYPE
from llm_pipeline.v2.utils import load_prompt
from llm_pipeline.common.io_utils import load_text
from llm_pipeline.common.metrics import MetricsLogger, asdict

def print_categories(result):
    """Print categories and confidence scores from the structured result."""
    print("\n=== Categories ===")
    if not isinstance(result, dict):
        print("Error: Result is not a dictionary")
        return
        
    categories = result.get('categories', {})
    if not categories:
        print("No categories found in result")
        return
        
    # Sort categories by confidence score in descending order
    sorted_categories = sorted(
        categories.items(),
        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
        reverse=True
    )
    
    # Print each category and its confidence score
    for category, confidence in sorted_categories:
        if isinstance(confidence, (int, float)):
            print(f"{category}: {confidence:.2%}")
        else:
            print(f"{category}: {confidence}")

def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test the LLM prompt with example text")
    parser.add_argument("--input-file", type=Path, help="Path to a text file containing example text")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated prompt and exit without calling the LLM")
    parser.add_argument("--output", type=Path, help="Path to save the LLM result as JSON and the prompt as text")
    parser.add_argument("--print-raw", action="store_true", help="Print the raw LLM API response before parsing")
    parser.add_argument("--categories", action="store_true", help="Print categories and confidence scores from the result")
    parser.add_argument("--api-type", choices=["llama", "mixtral"], default="llama",
                      help="Type of API to use (default: llama)")
    args = parser.parse_args()

    # Initialize metrics logger and start a new run
    metrics_logger = MetricsLogger()
    run_id = metrics_logger.start_run()

    try:
        # Get example text from file or use default
        if args.input_file:
            try:
                example_text = load_text(args.input_file)
            except Exception as e:
                print(f"Error reading input file: {str(e)}")
                return
        else:
            # Default example text
            example_text = """
            Welcome to Acme Corporation!
            
            For sales inquiries, please contact our sales team at +1 (800) 555-1234 or email sales@acme.com.
            Our technical support team is available 24/7 at +1 (888) 555-5678.
            
            Looking to join our team? Contact HR at +1 (877) 555-9012 for career opportunities.
            Our main office can be reached at (555) 123-4567.
            
            For press inquiries, please call +1 (866) 555-3456.
            To reach our IT department, dial extension 7890.
            
            Note: The number 12345 is not a valid contact number.
            """

        # Load the prompt template
        prompt_template = load_prompt()
        
        # Fill the template with our example text
        try:
            prompt = prompt_template.replace("{webpage_text}", example_text)
        except Exception as e:
            print(f"Error formatting prompt: {str(e)}")
            return
        
        # Print the full prompt
        print("\n=== Full Prompt ===")
        print(prompt)
        
        # If output path is specified, save the prompt
        if args.output:
            prompt_path = args.output.with_suffix('.prompt.txt')
            try:
                prompt_path.write_text(prompt)
                print(f"\nSaved prompt to: {prompt_path}")
            except Exception as e:
                print(f"\nError saving prompt: {str(e)}")
        
        # If dry run is enabled, exit here
        if args.dry_run:
            print("\n=== Dry Run Mode: Exiting before LLM API call ===")
            return
            
        print(f"\n=== Sending to {args.api_type.upper()} API ===")
        
        try:
            # Query the LLM and parse the response
            response = query_llm_v2(prompt, api_type=args.api_type)
            
            # Print raw response if requested
            if args.print_raw:
                print("\n=== Raw LLM Response ===")
                print(json.dumps(response, indent=2))
                print("\n=== Parsed Result ===")
            
            result = parse_structured_output(response)
            
            # Print the structured result
            print("\n=== LLM Response ===")
            print(json.dumps(result, indent=2))
            
            # Print categories if requested
            if args.categories:
                print_categories(result)
            
            # If output path is specified, save the result
            if args.output:
                try:
                    args.output.write_text(json.dumps(result, indent=2))
                    print(f"\nSaved result to: {args.output}")
                except Exception as e:
                    print(f"\nError saving result: {str(e)}")
                    
        except Exception as e:
            print(f"\nError calling API: {str(e)}")
            raise
            
    except Exception as e:
        print("\n=== Error ===")
        traceback.print_exc()
        
    finally:
        # End the run and get metrics
        try:
            run_metrics = metrics_logger.end_run()
            print("\n=== Run Metrics ===")
            print(json.dumps(asdict(run_metrics), indent=2))
        except Exception as e:
            print(f"\nError generating run metrics: {str(e)}")

if __name__ == "__main__":
    main() 