"""Main entry point for the v1 LLM pipeline."""

import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Set, Dict, Any, Optional
import time
import logging
import signal
import sys
from datetime import datetime
import json

from llm_pipeline.v1.llm_client import query_llm, parse_llm_output, load_prompt, set_metrics_logger
from llm_pipeline.config import (
    PAGES_DIR, OCR_DIR, OUTPUTS_V1, IMAGES_DIR,
    set_custom_data_dir
)
from llm_pipeline.common import load_text, save_json, setup_logger
from llm_pipeline.common.metrics import MetricsLogger, APICallMetrics

# Set up logger
logger = setup_logger("llm_pipeline.v1", level=logging.WARNING)  # Default to WARNING level

# Global flag for graceful shutdown
should_continue = True

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global should_continue
    logger.info(f"\nReceived signal {signum}. Initiating graceful shutdown...")
    should_continue = False

def register_signal_handlers():
    """Register handlers for termination signals."""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request

def save_partial_results(run_dir: Path, processed_files: List[Path], failed_files: List[Path]) -> None:
    """Save partial results when pipeline is interrupted.
    
    Args:
        run_dir: Path to the run directory
        processed_files: List of successfully processed files
        failed_files: List of files that failed processing
    """
    summary = {
        "status": "interrupted",
        "timestamp": datetime.now().isoformat(),
        "processed_files": [str(f) for f in processed_files],
        "failed_files": [str(f) for f in failed_files],
        "total_processed": len(processed_files),
        "total_failed": len(failed_files)
    }
    
    summary_file = run_dir / "interrupted_summary.json"
    save_json(summary_file, summary)
    logger.info(f"Saved partial results to {summary_file}")

def create_run_directory(run_name: Optional[str] = None) -> Path:
    """Create a new directory for this run with timestamp and optional name.
    
    Args:
        run_name: Optional name to include in the directory name
        
    Returns:
        Path to the run directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        # Clean the run name to be filesystem safe
        safe_name = "".join(c for c in run_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        dir_name = f"{safe_name}_{timestamp}"
    else:
        dir_name = f"run_{timestamp}"
    
    run_dir = OUTPUTS_V1 / dir_name
    
    # Create directory structure
    run_dir.mkdir(exist_ok=True, parents=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)
    
    return run_dir

def get_url_directory(run_dir: Path, hostname: str) -> Path:
    """Get or create the directory for a specific URL.
    
    Args:
        run_dir: Path to the run directory
        hostname: The hostname to create directory for
        
    Returns:
        Path to the URL-specific directory
    """
    # Sanitize hostname for filesystem
    safe_hostname = hostname.replace(".", "_")
    url_dir = run_dir / "outputs" / safe_hostname
    
    # Create URL-specific directories
    url_dir.mkdir(exist_ok=True, parents=True)
    (url_dir / "debug").mkdir(exist_ok=True)
    (url_dir / "raw").mkdir(exist_ok=True)
    
    return url_dir

def save_run_summary(run_dir: Path, summary: Dict[str, Any]) -> None:
    """Save the run summary to the run directory.
    
    Args:
        run_dir: Path to the run directory
        summary: Summary data to save
    """
    summary_file = run_dir / "summary.json"
    save_json(summary_file, summary)

def setup_run_logging(run_dir: Path) -> logging.Logger:
    """Set up logging for this run.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Configured logger instance
    """
    log_file = run_dir / "logs" / "debug.log"
    return setup_logger("llm_pipeline.v1", log_file=log_file)

def get_source_files(source: str) -> List[Path]:
    """
    Get files to process based on source type.
    
    Args:
        source: One of 'text', 'ocr', or 'ocr+text'
        
    Returns:
        List of file paths to process
    """
    global logger
    # Import here to get the updated path
    from llm_pipeline.config import PAGES_DIR
    
    logger.debug(f"Searching for files in PAGES_DIR: {PAGES_DIR}")
    logger.debug(f"PAGES_DIR exists: {PAGES_DIR.exists()}")
    
    if source == 'text':
        # Look for text.txt files in the pages directory
        pattern = "**/text.txt"
        logger.debug(f"Searching with pattern: {pattern}")
        files = list(PAGES_DIR.glob(pattern))
        logger.debug(f"Found {len(files)} text files")
        for file in files:
            logger.debug(f"Found text file: {file}")
        return files
    elif source == 'ocr':
        # Look for summary.json files in the ocr subdirectories
        pattern = "**/ocr/summary.json"
        logger.debug(f"Searching with pattern: {pattern}")
        files = list(PAGES_DIR.glob(pattern))
        logger.debug(f"Found {len(files)} OCR files")
        for file in files:
            logger.debug(f"Found OCR file: {file}")
        return files
    elif source == 'ocr+text':
        # Get both text and OCR files
        text_pattern = "**/text.txt"
        ocr_pattern = "**/ocr/summary.json"
        logger.debug(f"Searching with patterns: {text_pattern} and {ocr_pattern}")
        text_files = set(PAGES_DIR.glob(text_pattern))
        ocr_files = set(PAGES_DIR.glob(ocr_pattern))
        logger.debug(f"Found {len(text_files)} text files and {len(ocr_files)} OCR files")
        for file in text_files:
            logger.debug(f"Found text file: {file}")
        for file in ocr_files:
            logger.debug(f"Found OCR file: {file}")
        return list(text_files | ocr_files)
    else:
        raise ValueError(f"Invalid source type: {source}")

def get_hostname_from_path(text_file: Path) -> str:
    """Extract the hostname from the text file path."""
    # Handle both text.txt and ocr/summary.json paths
    if text_file.name == "summary.json":
        return text_file.parent.parent.name
    return text_file.parent.name

def process_file(file_path: Path, prompt_template: str, run_dir: Path, debug: bool = False, api_type: str = 'llama') -> dict:
    """Process a single file and return the LLM output."""
    logger.debug(f"Processing file: {file_path}")
    logger.debug(f"Debug mode: {debug}")
    logger.debug(f"API type: {api_type}")
    logger.debug(f"Run directory: {run_dir}")
    
    # Read the text content
    try:
        text_content = load_text(file_path)
        logger.debug(f"Successfully loaded text content from {file_path}")
        logger.debug(f"Text content length: {len(text_content)} characters")
        logger.debug(f"First 500 chars of content: {text_content[:500]}")
    except Exception as e:
        logger.error(f"Failed to load text content from {file_path}: {str(e)}")
        raise
    
    # Check for empty or very short content
    MIN_CONTENT_LENGTH = 50
    content_length = len(text_content.strip())
    logger.debug(f"Content length after stripping: {content_length}")
    
    if not text_content or content_length < MIN_CONTENT_LENGTH:
        hostname = get_hostname_from_path(file_path)
        logger.warning(f"Skipping {hostname}: Content too short ({content_length} chars) or empty")
        return {
            "phone_numbers": [],
            "confidence": 0.0,
            "error": "Content too short or empty"
        }
    
    # Clean and prepare the text content
    cleaned_text = text_content.strip()
    logger.debug(f"Cleaned text length: {len(cleaned_text)}")
    logger.debug(f"First 100 chars of cleaned text: {cleaned_text[:100]}")
    
    # Build the prompt
    prompt = prompt_template.format(webpage_text=cleaned_text)
    logger.debug(f"Built prompt with length: {len(prompt)}")
    
    # Get URL directory for this file
    hostname = get_hostname_from_path(file_path)
    url_dir = get_url_directory(run_dir, hostname)
    logger.debug(f"Using URL directory: {url_dir}")
    
    # Maximum number of retries for API calls
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Start timing the LLM query
            start_time = time.monotonic()
            logger.debug(f"Attempt {attempt + 1}/{max_retries}: Querying LLM...")
            logger.debug(f"Using API type: {api_type}")
            
            # Query the LLM and get raw response
            response = query_llm(prompt, api_type=api_type)
            logger.debug(f"Got response from LLM")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            logger.debug(f"Full response: {json.dumps(response, indent=2)}")
            
            # Calculate and log latency
            latency = time.monotonic() - start_time
            logger.debug(f"LLM latency for {hostname}: {latency:.2f}s")
            
            # Save raw response
            try:
                raw_output_file = url_dir / "raw" / "response.json"
                save_json(raw_output_file, response)
                logger.debug(f"Saved raw response to {raw_output_file}")
            except Exception as e:
                logger.error(f"Failed to save raw response: {str(e)}")
            
            if debug:
                logger.debug(f"Raw LLM response for {hostname}: {json.dumps(response, indent=2)}")
            
            # Parse the response
            logger.debug("Parsing LLM response...")
            result = parse_llm_output(response)
            logger.debug(f"Successfully parsed response")
            logger.debug(f"Parsed result type: {type(result)}")
            logger.debug(f"Parsed result: {json.dumps(result, indent=2)}")
            
            # Save the final output
            try:
                output_file = url_dir / "output.json"
                save_json(output_file, result)
                logger.debug(f"Saved final output to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save final output: {str(e)}")
            
            return result
            
        except Exception as e:
            last_error = e
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            logger.debug("Full traceback:", exc_info=True)
            if attempt < max_retries - 1:
                backoff_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Attempt {attempt + 1} failed for {file_path.name}. Retrying in {backoff_time}s. Error: {str(e)}")
                time.sleep(backoff_time)
            else:
                logger.error(f"All {max_retries} attempts failed for {file_path.name}. Last error: {str(e)}")
                raise last_error

def main() -> None:
    """Main entry point for the v1 pipeline."""
    parser = argparse.ArgumentParser(description="Process text files through the v1 LLM pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--api-type", choices=["llama", "mixtral"], default="llama",
                      help="Type of API to use (default: llama)")
    parser.add_argument("--data-dir", type=Path, help="Custom data directory path")
    parser.add_argument("--run-name", type=str, help="Optional name for this run")
    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        # In non-debug mode, only show warnings and errors
        logger.setLevel(logging.WARNING)

    logger.debug("Starting v1 pipeline with arguments:")
    logger.debug(f"Debug mode: {args.debug}")
    logger.debug(f"API type: {args.api_type}")
    logger.debug(f"Data directory: {args.data_dir}")
    logger.debug(f"Run name: {args.run_name}")

    # Register signal handlers for graceful shutdown
    register_signal_handlers()
    logger.debug("Registered signal handlers")
    
    # Set up custom data directory if provided
    if args.data_dir:
        data_dir = Path(args.data_dir)
        logger.debug(f"Processing data directory: {data_dir}")
        logger.debug(f"Data directory exists: {data_dir.exists()}")
        logger.debug(f"Data directory is directory: {data_dir.is_dir()}")
        
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return
        if not data_dir.is_dir():
            logger.error(f"Data directory path is not a directory: {data_dir}")
            return
        try:
            # Test if directory is readable
            next(data_dir.iterdir())
            logger.debug("Successfully verified data directory is readable")
        except (PermissionError, StopIteration) as e:
            logger.error(f"Error accessing data directory {data_dir}: {str(e)}")
            return
            
        logger.info(f"Using custom data directory: {data_dir}")
        set_custom_data_dir(data_dir)
        
        # Update PAGES_DIR to point to the correct location
        from llm_pipeline.config import PAGES_DIR
        logger.debug(f"Updated PAGES_DIR to: {PAGES_DIR}")
        logger.debug(f"PAGES_DIR exists: {PAGES_DIR.exists()}")
        logger.debug(f"PAGES_DIR is directory: {PAGES_DIR.is_dir()}")
    
    # Create run directory
    try:
        run_dir = create_run_directory(args.run_name)
        logger.debug(f"Created run directory: {run_dir}")
        logger.debug(f"Run directory structure:")
        for item in run_dir.rglob("*"):
            logger.debug(f"  {item.relative_to(run_dir)}")
    except Exception as e:
        logger.error(f"Failed to create run directory: {str(e)}")
        return
    
    # Set up metrics logger
    try:
        metrics_logger = MetricsLogger()
        metrics_logger.start_run()  # Start the run before setting the logger
        set_metrics_logger(metrics_logger)
        logger.debug("Successfully set up metrics logger")
    except Exception as e:
        logger.error(f"Failed to set up metrics logger: {str(e)}")
        return
    
    # Load the prompt template
    try:
        prompt_template = load_prompt()
        logger.debug("Successfully loaded prompt template")
        logger.debug(f"Prompt template length: {len(prompt_template)} characters")
        logger.debug(f"First 500 chars of prompt template: {prompt_template[:500]}")
    except Exception as e:
        logger.error(f"Failed to load prompt template: {str(e)}")
        return
    
    # Get list of files to process
    text_files = []
    try:
        logger.debug(f"Searching for text files in {PAGES_DIR}")
        # Use rglob to find text.txt files in all subdirectories
        for text_file in PAGES_DIR.rglob("text.txt"):
            if text_file.is_file():
                text_files.append(text_file)
                logger.debug(f"Found text file: {text_file}")
                logger.debug(f"File size: {text_file.stat().st_size} bytes")
                logger.debug(f"Last modified: {datetime.fromtimestamp(text_file.stat().st_mtime)}")
    except Exception as e:
        logger.error(f"Error searching for text files: {str(e)}")
        return
    
    if not text_files:
        logger.error(f"No text files found in {PAGES_DIR}")
        logger.debug("Directory contents:")
        try:
            for item in PAGES_DIR.iterdir():
                logger.debug(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
                if item.is_file():
                    logger.debug(f"    Size: {item.stat().st_size} bytes")
                    logger.debug(f"    Last modified: {datetime.fromtimestamp(item.stat().st_mtime)}")
        except Exception as e:
            logger.error(f"Error listing directory contents: {str(e)}")
        return
    
    logger.info(f"Found {len(text_files)} files to process")
    logger.debug("Files to process:")
    for file in text_files:
        logger.debug(f"  {file}")
    
    # Track processed and failed files
    processed_files = []
    failed_files = []
    
    try:
        # Process each file with progress bar
        for file_path in tqdm(text_files, desc="Processing files"):
            if not should_continue:
                logger.info("Graceful shutdown requested. Saving partial results...")
                break
                
            try:
                logger.debug(f"\nProcessing file: {file_path}")
                result = process_file(file_path, prompt_template, run_dir, args.debug, args.api_type)
                processed_files.append(file_path)
                # Log successful processing to metrics
                if metrics_logger:
                    metrics_logger.log_api_call(
                        APICallMetrics(
                            timestamp=datetime.now().isoformat(),
                            api_type=args.api_type,
                            prompt_length=len(prompt_template),
                            response_length=len(str(result)),
                            total_duration=0.0,  # This will be updated by the actual API call
                            success=True
                        ),
                        url=str(file_path)
                    )
                logger.debug(f"Successfully processed {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                failed_files.append(file_path)
                # Log failed processing to metrics
                if metrics_logger:
                    metrics_logger.log_api_call(
                        APICallMetrics(
                            timestamp=datetime.now().isoformat(),
                            api_type=args.api_type,
                            prompt_length=len(prompt_template),
                            response_length=0,
                            total_duration=0.0,
                            success=False,
                            error_message=str(e)
                        ),
                        url=str(file_path)
                    )
                if args.debug:
                    logger.debug("Full traceback:", exc_info=True)
    
    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt. Saving partial results...")
    finally:
        # Save final summary
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "run_name": args.run_name,
                "total_files": len(text_files),
                "processed_files": [str(f) for f in processed_files],
                "failed_files": [str(f) for f in failed_files],
                "total_processed": len(processed_files),
                "total_failed": len(failed_files),
                "api_type": args.api_type,
                "debug_mode": args.debug
            }
            save_run_summary(run_dir, summary)
            logger.info(f"Saved run summary to {run_dir}/summary.json")
        except Exception as e:
            logger.error(f"Failed to save run summary: {str(e)}")
        
        # End metrics logging
        try:
            run_metrics = metrics_logger.end_run()
            metrics_file = run_dir / "metrics.json"
            # Convert RunMetrics dataclass to dictionary before saving
            metrics_dict = {
                "run_id": run_metrics.run_id,
                "start_time": run_metrics.start_time,
                "end_time": run_metrics.end_time,
                "total_urls": run_metrics.total_urls,
                "successful_urls": run_metrics.successful_urls,
                "failed_urls": run_metrics.failed_urls,
                "total_api_calls": run_metrics.total_api_calls,
                "successful_api_calls": run_metrics.successful_api_calls,
                "failed_api_calls": run_metrics.failed_api_calls,
                "total_duration": run_metrics.total_duration,
                "avg_duration_per_url": run_metrics.avg_duration_per_url,
                "avg_duration_per_call": run_metrics.avg_duration_per_call,
                "calls_by_api": run_metrics.calls_by_api,
                "errors": run_metrics.errors,
                "urls_processed": run_metrics.urls_processed
            }
            save_json(metrics_file, metrics_dict)
            logger.info(f"Saved metrics to {metrics_file}")
            logger.debug(f"Metrics summary: {json.dumps(metrics_dict, indent=2)}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
        
        # Log final status
        logger.info(f"\nProcessing complete:")
        logger.info(f"- Successfully processed: {len(processed_files)} files")
        logger.info(f"- Failed: {len(failed_files)} files")
        if failed_files:
            logger.info("Failed files:")
            for f in failed_files:
                logger.info(f"  - {f}")

if __name__ == "__main__":
    main() 