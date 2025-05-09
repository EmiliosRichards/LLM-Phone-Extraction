"""Unified main controller for phone number extraction pipeline.

This script combines the functionality of the scraper and LLM pipeline to:
1. Scrape web pages for text and images
2. Process the extracted content through the LLM pipeline
3. Run both components concurrently when possible
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import time
import subprocess
import sys

from scraper import scrape_page, config as scraper_config
from scraper.main import ScrapingSession, process_url, ScrapingError
from scraper.utils import normalize_hostname

# Import LLM pipeline modules
from llm_pipeline.v1.llm_client import query_llm as query_llm_v1, parse_structured_output as parse_structured_output_v1
from llm_pipeline.v1.utils import get_text_files as get_text_files_v1, get_hostname_from_path as get_hostname_from_path_v1, load_prompt as load_prompt_v1
from llm_pipeline.v2.llm_client import query_llm_v2, parse_structured_output as parse_structured_output_v2
from llm_pipeline.v2.utils import get_text_files as get_text_files_v2, get_hostname_from_path as get_hostname_from_path_v2, load_prompt as load_prompt_v2
from llm_pipeline.config import PAGES_DIR, OCR_DIR, OUTPUTS_V1, OUTPUTS_V2
from llm_pipeline.common import load_text, save_json, setup_logger

# Set up logging
logger = setup_logger("main_controller")

class UnifiedPipeline:
    """Combines scraper and LLM pipeline functionality."""
    
    def __init__(self, run_name: Optional[str] = None, version: str = 'v2', api_type: str = 'openai'):
        """Initialize the unified pipeline.
        
        Args:
            run_name: Optional name for this run
            version: Which version of the LLM pipeline to use ('v1' or 'v2')
            api_type: Which LLM API to use ('openai', 'anthropic', 'gemini', etc.)
        """
        self.run_name = run_name
        self.version = version
        self.api_type = api_type
        self.start_time = datetime.now()
        self.failed_urls = []  # Track failed URLs
        
        # Create run directories
        self.llm_run_dir = self._create_llm_run_directory(run_name)
        
        # Load LLM prompt and set up version-specific functions
        if version == 'v1':
            self.prompt_template = load_prompt_v1()
            self.query_llm = query_llm_v1
            self.parse_structured_output = parse_structured_output_v1
            self.get_text_files = get_text_files_v1
            self.get_hostname_from_path = get_hostname_from_path_v1
        else:  # v2
            self.prompt_template = load_prompt_v2()
            self.query_llm = query_llm_v2
            self.parse_structured_output = parse_structured_output_v2
            self.get_text_files = get_text_files_v2
            self.get_hostname_from_path = get_hostname_from_path_v2
        
        logger.info(f"Initialized unified pipeline with run name: {run_name} (LLM Pipeline {version}, API: {api_type})")
        logger.info(f"LLM run directory: {self.llm_run_dir}")
    
    def _create_llm_run_directory(self, run_name: Optional[str] = None) -> Path:
        """Create a new directory for LLM pipeline run.
        
        Args:
            run_name: Optional name to include in the directory name
            
        Returns:
            Path to the run directory
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if run_name:
            safe_name = "".join(c for c in run_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            dir_name = f"{safe_name}_{timestamp}"
        else:
            dir_name = f"run_{timestamp}"
        
        # Use version-specific output directory
        base_dir = OUTPUTS_V1 if self.version == 'v1' else OUTPUTS_V2
        run_dir = base_dir / dir_name
        run_dir.mkdir(exist_ok=True, parents=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "outputs").mkdir(exist_ok=True)
        
        return run_dir
    
    def _verify_scraping_result(self, hostname: str, max_retries: int = 3, retry_delay: float = 1.0) -> Tuple[bool, List[Path]]:
        """Verify that scraping completed successfully by checking for text files.
        
        Args:
            hostname: The hostname to check
            max_retries: Maximum number of retries to check for files
            retry_delay: Delay between retries in seconds
            
        Returns:
            Tuple of (success, list of text files found)
        """
        for attempt in range(max_retries):
            text_files = list(PAGES_DIR.glob(f"{hostname}/**/text.txt"))
            if text_files:
                return True, text_files
            
            if attempt < max_retries - 1:
                logger.warning(f"No text files found for {hostname} (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        return False, []
    
    def _get_unique_output_path(self, hostname: str, text_file: Path) -> Path:
        """Generate a unique output path for a text file.
        
        Args:
            hostname: The hostname being processed
            text_file: The text file being processed
            
        Returns:
            Path for the output file
        """
        # Get the relative path of the text file from the hostname directory
        rel_path = text_file.relative_to(PAGES_DIR / hostname)
        
        # Create a unique name based on the path and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_path = str(rel_path).replace('/', '_').replace('\\', '_')
        output_name = f"output_{safe_path}_{timestamp}.json"
        
        # Create the output directory structure
        output_dir = self.llm_run_dir / hostname
        output_dir.mkdir(exist_ok=True, parents=True)
        
        return output_dir / output_name
    
    async def process_url(self, url: str, scrape_mode: str = 'both') -> None:
        """Process a single URL through both scraper and LLM pipeline.
        
        Args:
            url: The URL to process
            scrape_mode: What to scrape - 'text', 'ocr', or 'both'
        """
        hostname = normalize_hostname(url)
        scraping_success = False
        
        try:
            # First run the scraper as a separate process
            logger.info(f"Starting scraping for {url}")
            cmd = [
                sys.executable, "-m", "scraper.main",
                "--url", url,
                "--scrape-mode", scrape_mode,
                "--log-level", "INFO"
            ]
            if self.run_name:
                cmd.extend(["--run-name", self.run_name])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Scraper failed: {result.stderr}")
            
            # Verify scraping completed successfully
            scraping_success, text_files = self._verify_scraping_result(hostname)
            if not scraping_success:
                raise Exception(f"Scraping completed but no text files found for {url}")
            
            logger.info(f"Scraping completed successfully for {url}, found {len(text_files)} text files")
            
            # Process through LLM pipeline
            for text_file in text_files:
                try:
                    # Read the text content
                    text_content = load_text(text_file)
                    
                    # Skip empty or very short content
                    if not text_content or len(text_content.strip()) < 50:
                        logger.warning(f"Skipping {text_file}: Content too short or empty")
                        continue
                    
                    # Build the prompt
                    prompt = self.prompt_template.format(webpage_text=text_content)
                    
                    # Query the LLM
                    logger.info(f"Processing {text_file} through LLM pipeline using {self.api_type} API")
                    response = await asyncio.to_thread(self.query_llm, prompt, api_type=self.api_type)
                    
                    # Parse the response
                    result = self.parse_structured_output(response)
                    
                    # Generate unique output path and save result
                    output_file = self._get_unique_output_path(hostname, text_file)
                    save_json(output_file, result)
                    
                    logger.info(f"Successfully processed {text_file} through LLM pipeline, saved to {output_file}")
                    
                except Exception as e:
                    logger.error(f"Error processing {text_file} through LLM pipeline: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            self.failed_urls.append((url, str(e)))
        finally:
            # Update scraping session metrics
            summary = {
                'timestamp': {
                    'start': self.start_time.isoformat(),
                    'end': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - self.start_time).total_seconds()
                },
                'extraction': {
                    'metrics': {
                        'ocr_attempts': 0,
                        'ocr_successes': 0
                    }
                }
            }
            self.scraping_session.add_url_result(url, summary, scraping_success)
    
    async def process_urls(self, urls: List[str], scrape_mode: str = 'both') -> None:
        """Process multiple URLs concurrently.
        
        Args:
            urls: List of URLs to process
            scrape_mode: What to scrape - 'text', 'ocr', or 'both'
        """
        # Process URLs concurrently with a semaphore to limit concurrency
        sem = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
        
        async def process_with_semaphore(url: str):
            try:
                async with sem:
                    await self.process_url(url, scrape_mode)
            except Exception as e:
                logger.error(f"Unexpected error in process_with_semaphore for {url}: {str(e)}")
                self.failed_urls.append((url, str(e)))
        
        tasks = [process_with_semaphore(url) for url in urls]
        await asyncio.gather(*tasks)
        
        # Log summary of failed URLs
        if self.failed_urls:
            logger.warning(f"\nFailed URLs ({len(self.failed_urls)}):")
            for url, error in self.failed_urls:
                logger.warning(f"- {url}: {error}")
    
    def save_summary(self) -> None:
        """Save summary of the pipeline run."""
        # Get scraping session summary
        scraping_summary = self.scraping_session.get_session_summary()
        
        # Create unified summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "run_name": self.run_name,
            "scraping": scraping_summary,
            "llm_pipeline": {
                "run_directory": str(self.llm_run_dir),
                "prompt_template": str(self.prompt_template)
            },
            "failed_urls": [
                {"url": url, "error": error}
                for url, error in self.failed_urls
            ]
        }
        
        # Save summary
        summary_file = self.llm_run_dir / "summary.json"
        save_json(summary_file, summary)
        logger.info(f"Saved pipeline summary to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Unified phone number extraction pipeline")
    parser.add_argument('--url', help='Single URL to process')
    parser.add_argument('--url-file', help='Path to file containing URLs to process (one per line)')
    parser.add_argument('--scrape-mode', choices=['text', 'ocr', 'both'], default='both',
                      help='What to scrape: text only, OCR only, or both (default: both)')
    parser.add_argument('--run-name', help='Name for this pipeline run')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    parser.add_argument('--version', choices=['v1', 'v2'], default='v2',
                      help='Which version of the LLM pipeline to use (default: v2)')
    parser.add_argument('--api-type', choices=['openai', 'anthropic', 'gemini'], default='openai',
                      help='Which LLM API to use (default: openai)')
    args = parser.parse_args()
    
    # Set up logging
    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level))
    
    # Initialize pipeline
    pipeline = UnifiedPipeline(args.run_name, args.version, args.api_type)
    
    # Get URLs to process
    urls = []
    if args.url:
        urls.append(args.url)
    elif args.url_file:
        with open(args.url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        logger.error("No URLs provided. Use --url or --url-file")
        return
    
    # Run the pipeline
    try:
        asyncio.run(pipeline.process_urls(urls, args.scrape_mode))
        pipeline.save_summary()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 