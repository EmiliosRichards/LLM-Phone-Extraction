"""Test script for v1 prompt."""

import argparse
import json
from pathlib import Path
from dataclasses import asdict
from llm_pipeline.v1.llm_client import query_llm, load_prompt
from llm_pipeline.common.metrics import MetricsLogger

def main() -> None:
    """Run the test prompt."""
    parser = argparse.ArgumentParser(description="Test the v1 prompt with different API types")
    parser.add_argument("--api-type", choices=["llama", "mixtral"], default="llama",
                      help="Type of API to use (default: llama)")
    args = parser.parse_args()
    
    # Initialize metrics logger and start a new run
    metrics_logger = MetricsLogger()
    run_id = metrics_logger.start_run()
    
    try:
        # Load the prompt
        prompt = load_prompt()
        print("\n=== Full Prompt ===")
        print(prompt)
        
        # Send to API
        print(f"\n=== Sending to {args.api_type.upper()} API ===")
        try:
            response = query_llm(prompt, api_type=args.api_type)
            
            # Parse and display the response
            print("\n=== Parsed LLM Response ===")
            if isinstance(response, dict):
                print(json.dumps(response, indent=2))
            else:
                print(response)
        except Exception as e:
            print(f"\nError calling API: {str(e)}")
            raise
            
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