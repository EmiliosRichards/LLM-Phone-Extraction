"""Main entry point for the v2 LLM pipeline."""

import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Set, Dict, Any
import time
from datetime import datetime

from llm_pipeline.v2.llm_client import query_llm_v2, parse_structured_output
from llm_pipeline.v2.utils import get_text_files, get_hostname_from_path, load_prompt
from llm_pipeline.config import PAGES_DIR, OCR_DIR, OUTPUTS_V2, set_custom_data_dir
from llm_pipeline.common import load_text, save_json, setup_logger

# Set up logger
logger = setup_logger("llm_pipeline.v2")

# Create debug directory
DEBUG_DIR = OUTPUTS_V2 / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

# Create logs directory
LOGS_DIR = OUTPUTS_V2 / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def get_source_files(source: str) -> List[Path]:
    """
    Get files to process based on source type.
    
    Args:
        source: One of 'text', 'ocr', or 'ocr+text'
        
    Returns:
        List of file paths to process
    """
    if source == 'text':
        return list(PAGES_DIR.glob("**/text.txt"))
    elif source == 'ocr':
        return list(PAGES_DIR.glob("**/ocr/summary.json"))
    elif source == 'ocr+text':
        # Get both text and OCR files
        text_files = set(PAGES_DIR.glob("**/text.txt"))
        ocr_files = set(PAGES_DIR.glob("**/ocr/summary.json"))
        return list(text_files | ocr_files)
    else:
        raise ValueError(f"Invalid source type: {source}")

def get_hostname_from_path(text_file: Path) -> str:
    """Extract the hostname from the text file path."""
    # Handle both text.txt and ocr/summary.json paths
    if text_file.name == "summary.json":
        return text_file.parent.parent.name
    return text_file.parent.name

def process_file(file_path: Path, prompt_template: str, debug: bool = False, api_type: str = "llama") -> Dict[str, Any]:
    """Process a single file and return the structured LLM output."""
    # Read the text content
    text_content = load_text(file_path)
    
    # Skip processing if text is too short
    if not text_content or len(text_content) < 50:
        hostname = get_hostname_from_path(file_path)
        logger.warning(f"Skipping {hostname}: Input text too short ({len(text_content)} chars)")
        return {
            "phone_numbers": [],
            "metadata": {
                "skipped": True,
                "reason": "Input text too short",
                "text_length": len(text_content)
            }
        }
    
    # Build the prompt by formatting the template
    prompt = prompt_template.format(webpage_text=text_content)
    
    # Query the LLM and parse the response
    start_time = time.monotonic()
    response = query_llm_v2(prompt, api_type=api_type)
    duration = time.monotonic() - start_time
    
    hostname = get_hostname_from_path(file_path)
    logger.debug(f"LLM query latency for {hostname}: {duration:.2f} seconds")
    
    # Save debug outputs if enabled
    if debug:
        debug_dir = DEBUG_DIR / hostname
        debug_dir.mkdir(exist_ok=True)
        
        # Save prompt
        with open(debug_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
            
        # Save raw response
        with open(debug_dir / "response.txt", "w", encoding="utf-8") as f:
            f.write(str(response))
    
    result = parse_structured_output(response)
    
    return result

def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process files for structured phone number extraction")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess already completed hostnames")
    parser.add_argument("--source", choices=['text', 'ocr', 'ocr+text'], default='text',
                      help="Source type to process (default: text)")
    parser.add_argument("--debug", action="store_true", help="Save debug outputs including prompts and raw responses")
    parser.add_argument("--hostnames", type=str, help="Comma-separated list of hostnames to process")
    parser.add_argument("--preview", action="store_true", help="Print the first processed result")
    parser.add_argument("--api-type", choices=['llama', 'mixtral'], default='llama',
                      help="Type of LLM API to use (default: llama)")
    parser.add_argument("--data-dir", type=str,
                      help="Path to the custom data directory containing pages, images, and ocr_text subdirectories")
    args = parser.parse_args()

    # Set custom data directory if specified
    if args.data_dir:
        set_custom_data_dir(Path(args.data_dir))
        logger.info(f"Using custom data directory: {args.data_dir}")

    # Create output directory if it doesn't exist
    OUTPUTS_V2.mkdir(exist_ok=True)
    
    # Load the prompt template
    prompt_template = load_prompt()
    
    # Get files to process based on source type
    try:
        files = get_source_files(args.source)
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Filter by hostnames if specified
    if args.hostnames:
        target_hostnames = {h.strip() for h in args.hostnames.split(",")}
        files = [f for f in files if get_hostname_from_path(f) in target_hostnames]
        if not files:
            logger.error(f"No files found for specified hostnames: {', '.join(target_hostnames)}")
            return
        logger.info(f"Filtered to {len(files)} files for {len(target_hostnames)} hostnames")
    
    # Apply limit if specified
    if args.limit:
        files = files[:args.limit]
    
    # Track statistics
    total_files = len(files)
    processed_files = 0
    skipped_files = 0
    error_files = 0
    
    # Process each file with a progress bar
    for file_path in tqdm(files, desc="Processing files"):
        hostname = get_hostname_from_path(file_path)
        output_file = OUTPUTS_V2 / f"{hostname}.json"
        
        # Skip if already processed and not overwriting
        if output_file.exists() and not args.overwrite:
            skipped_files += 1
            continue
        
        try:
            # Process the file
            result = process_file(file_path, prompt_template, debug=args.debug, api_type=args.api_type)
            
            # Save the result
            save_json(output_file, result)
            
            processed_files += 1
            logger.info(f"Processed {hostname}")
            
            # Print preview of first processed result
            if args.preview and processed_files == 1:
                logger.info("\nPreview of first processed result:")
                logger.info(f"Hostname: {hostname}")
                logger.info("Result:")
                logger.info(f"Phone numbers: {result.get('phone_numbers', [])}")
                logger.info(f"Metadata: {result.get('metadata', {})}")
                logger.info("---")
                
        except Exception as e:
            logger.error(f"Error processing {hostname}: {str(e)}")
            error_files += 1
            continue
    
    # Print summary
    logger.info("\nProcessing Summary:")
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Files processed: {processed_files}")
    logger.info(f"Files skipped: {skipped_files}")
    logger.info(f"Files with errors: {error_files}")
    
    # Save summary to JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_files": total_files,
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "error_files": error_files,
        "source": args.source,
        "hostnames_filter": args.hostnames if args.hostnames else None,
        "limit": args.limit,
        "overwrite": args.overwrite,
        "debug": args.debug,
        "api_type": args.api_type
    }
    
    # Save with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = LOGS_DIR / f"summary_{timestamp}.json"
    save_json(summary_file, summary)
    logger.info(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main() 