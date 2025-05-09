import argparse
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Iterator, Dict, Any, Tuple
from urllib.parse import urlparse, ParseResult
import requests
from requests.exceptions import RequestException
import sys
import psutil

# Check if this module is being imported by main.py
if 'main' in sys.modules and 'scraper.main' in sys.modules['main'].__file__:
    raise ImportError("The scraper module cannot be imported by main.py. Please run it directly.")

from . import config
from .scraper import scrape_page
from scraper.utils import (
    validate_url,
    create_metadata,
    normalize_hostname
)

class ScrapingError(Exception):
    """Base exception for scraping errors."""
    def __init__(self, message: str, error_type: str = "Unknown", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now()

class InvalidURLError(ScrapingError):
    """Raised when the URL is invalid."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "InvalidURL", details)

class ConnectionError(ScrapingError):
    """Raised when there are connection issues."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "Connection", details)

class ParsingError(ScrapingError):
    """Raised when there are issues parsing the page content."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "Parsing", details)

class OCRError(ScrapingError):
    """Raised when there are issues with OCR processing."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "OCR", details)

class ScrapingSession:
    """Tracks metrics across multiple URLs in a scraping session."""
    
    def __init__(self):
        self.total_urls = 0
        self.total_time = 0.0
        self.total_ocr_attempts = 0
        self.total_ocr_successes = 0
        self.start_time = datetime.now()
        self.failed_urls = []  # List of tuples (url, error)
        self.successful_urls = []
    
    def add_url_result(self, url: str, summary: Dict[str, Any], success: bool, error: Optional[ScrapingError] = None) -> None:
        """Add results from a single URL to the session metrics.
        
        Args:
            url: The URL that was processed
            summary: The scraping summary for this URL
            success: Whether the URL was processed successfully
            error: The error that occurred, if any
        """
        self.total_urls += 1
        
        # Add timing
        self.total_time += summary['timestamp']['duration_seconds']
        
        # Add OCR metrics
        metrics = summary['extraction']['metrics']
        self.total_ocr_attempts += metrics['ocr_attempts']
        self.total_ocr_successes += metrics['ocr_successes']
        
        # Track URL status
        if success:
            self.successful_urls.append(url)
        else:
            self.failed_urls.append((url, error))
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Generate a summary of the entire scraping session.
        
        Returns:
            Dict containing session-wide metrics and statistics
        """
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate average OCR success rate
        avg_ocr_success_rate = 0.0
        if self.total_ocr_attempts > 0:
            avg_ocr_success_rate = (self.total_ocr_successes / self.total_ocr_attempts) * 100
        
        return {
            'session_duration': {
                'start': self.start_time.isoformat(),
                'end': end_time.isoformat(),
                'total_seconds': total_duration
            },
            'urls_processed': {
                'total': self.total_urls,
                'successful': len(self.successful_urls),
                'failed': len(self.failed_urls)
            },
            'ocr_metrics': {
                'total_attempts': self.total_ocr_attempts,
                'total_successes': self.total_ocr_successes,
                'average_success_rate': round(avg_ocr_success_rate, 2)
            },
            'performance': {
                'total_processing_time': round(self.total_time, 2),
                'average_time_per_url': round(self.total_time / self.total_urls if self.total_urls > 0 else 0, 2)
            }
        }

def get_output_paths(base_dir: Path, url: str, content_type: str = 'text') -> Dict[str, Path]:
    """Generate output paths with appropriate extensions based on content type.
    
    Args:
        base_dir: Base directory for output files
        url: The URL being scraped (used for filename)
        content_type: Type of content ('text', 'json', 'html', etc.)
        
    Returns:
        Dict mapping file types to their paths. Note: Directories are not created here.
        They should be created only when content is ready to be saved.
    """
    # Create a safe filename from the URL
    hostname = normalize_hostname(url)
    
    # Define extensions based on content type
    extensions = {
        'text': '.txt',
        'json': '.json',
        'html': '.html',
        'raw': '.raw',
        'ocr': '.ocr.txt',
        'ocr_summary': '.ocr.json'
    }
    
    # Get the current run directory
    run_dir = config.get_run_directory()
    
    # Generate paths with appropriate extensions
    paths = {
        'page': run_dir / 'pages' / hostname / f"page{extensions['html']}",
        'text': run_dir / 'pages' / hostname / f"text{extensions['text']}",
        'raw_text': run_dir / 'pages' / hostname / f"raw{extensions['raw']}",
        'images_dir': run_dir / 'images' / hostname,
        'ocr_dir': run_dir / 'pages' / hostname / "ocr",
        'ocr_summary': run_dir / 'pages' / hostname / "ocr" / f"summary{extensions['ocr_summary']}"
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
    
    return paths

def generate_scraping_summary(url: str, result: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
    """Generate a structured summary of the scraping results.
    
    Args:
        url: The URL that was scraped
        result: The scraping result dictionary
        start_time: When the scraping started
        
    Returns:
        Dict containing the structured summary
    """
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Generate output paths with appropriate extensions
    output_paths = get_output_paths(config.DATA_DIR, url)
    
    # Create base directory for files that will always exist
    output_paths['page'].parent.mkdir(parents=True, exist_ok=True)
    
    # Extract image statistics
    image_stats = {
        'total': len(result['images']),
        'by_type': {},
        'total_size_bytes': 0,
        'extensions': set(),  # Will be converted to list before returning
        'ocr_attempts': 0,
        'ocr_successes': 0,
        'ocr_success_rate': 0.0  # Initialize success rate
    }
    
    ocr_success_count = 0
    for img in result['images']:
        # Count image types and collect extensions
        img_type = img.get('image_type', 'unknown')
        image_stats['by_type'][img_type] = image_stats['by_type'].get(img_type, 0) + 1
        if 'extension' in img:
            image_stats['extensions'].add(img['extension'])
        
        # Sum up image sizes
        image_stats['total_size_bytes'] += img.get('size_bytes', 0)
        
        # Track OCR attempts and successes
        if img.get('ocr_attempted', False):
            image_stats['ocr_attempts'] += 1
            if img.get('ocr_text'):
                ocr_success_count += 1
                image_stats['ocr_successes'] += 1
    
    # Calculate OCR success rate
    if image_stats['ocr_attempts'] > 0:
        image_stats['ocr_success_rate'] = (image_stats['ocr_successes'] / image_stats['ocr_attempts']) * 100
    
    if image_stats['total'] > 0:
        # Create images directory only if we have images
        output_paths['images_dir'].mkdir(parents=True, exist_ok=True)
        
        # Create OCR directory only if we have OCR results
        if ocr_success_count > 0:
            output_paths['ocr_dir'].mkdir(parents=True, exist_ok=True)
    
    # Convert extensions set to sorted list for JSON serialization
    image_stats['extensions'] = sorted(list(image_stats['extensions']))
    
    # Generate text statistics
    text_stats = {
        'length': result['text_data']['text_length'],
        'word_count': result['text_data']['word_count'],
        'paragraph_count': result['text_data'].get('paragraph_count', 0),
        'has_content': bool(result['text'].strip()),
        'format': result.get('text_format', 'plain')
    }
    
    # Create the summary
    summary = {
        'timestamp': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'duration_seconds': duration
        },
        'url': {
            'original': url,
            'parsed': urlparse(url).geturl()
        },
        'extraction': {
            'success': bool(result),
            'text': text_stats,
            'images': image_stats,
            'metrics': {
                'total_time_seconds': duration,
                'ocr_attempts': image_stats['ocr_attempts'],
                'ocr_successes': image_stats['ocr_successes'],
                'ocr_success_rate': round(image_stats['ocr_success_rate'], 2)  # Round to 2 decimal places
            }
        },
        'output_files': {
            'page': str(output_paths['page']),
            'text': str(output_paths['text']),
            'raw_text': str(output_paths['raw_text']),
            'images_dir': str(output_paths['images_dir']) if image_stats['total'] > 0 else None,
            'ocr_dir': str(output_paths['ocr_dir']) if ocr_success_count > 0 else None,
            'ocr_summary': str(output_paths['ocr_summary']) if ocr_success_count > 0 else None
        }
    }
    
    return summary

def log_scraping_summary(summary: Dict[str, Any]) -> None:
    """Log the scraping summary in both human-readable and JSON formats.
    
    Args:
        summary: The structured summary dictionary
    """
    # Log human-readable summary
    logging.info("\n\n[SUMMARY] Scraping Result Summary:")
    logging.info(f"URL: {summary['url']['original']}")
    logging.info(f"Duration: {summary['timestamp']['duration_seconds']:.2f} seconds")
    
    # Text statistics
    text_stats = summary['extraction']['text']
    logging.info("\n[TEXT] Text Statistics:")
    logging.info(f"• Length: {text_stats['length']} characters")
    logging.info(f"• Words: {text_stats['word_count']} words")
    logging.info(f"• Paragraphs: {text_stats['paragraph_count']} paragraphs")
    logging.info(f"• Format: {text_stats['format']}")
    logging.info(f"• Has content: {'Yes' if text_stats['has_content'] else 'No'}")
    
    # Image statistics
    img_stats = summary['extraction']['images']
    logging.info("\n[IMAGES] Image Statistics:")
    logging.info(f"• Total images: {img_stats['total']}")
    logging.info(f"• Total size: {img_stats['total_size_bytes'] / 1024:.2f} KB")
    logging.info(f"• OCR success rate: {img_stats['ocr_success_rate']:.1f}%")
    if img_stats['by_type']:
        logging.info("• Types:")
        for img_type, count in img_stats['by_type'].items():
            logging.info(f"  - {img_type}: {count}")
    if img_stats['extensions']:
        logging.info("• File extensions:")
        for ext in sorted(img_stats['extensions']):
            logging.info(f"  - {ext}")
    
    # Performance metrics
    metrics = summary['extraction']['metrics']
    logging.info("\n[METRICS] Performance Metrics:")
    logging.info(f"• Total time: {metrics['total_time_seconds']:.2f} seconds")
    logging.info(f"• OCR attempts: {metrics['ocr_attempts']}")
    logging.info(f"• OCR successes: {metrics['ocr_successes']}")
    logging.info(f"• OCR success rate: {metrics['ocr_success_rate']:.1f}%")
    
    # Output files
    logging.info("\n[SAVED] Output Files:")
    for key, path in summary['output_files'].items():
        logging.info(f"• {key}: {path}")
    
    # Log JSON summary for machine parsing
    logging.info("\n[STATS] JSON Summary:")
    logging.info(json.dumps(summary, indent=2))

def log_session_summary(session: ScrapingSession) -> None:
    """Log the session-wide summary in a human-readable format.
    
    Args:
        session: The ScrapingSession object containing session metrics
    """
    summary = session.get_session_summary()
    
    logging.info("\n\n[SESSION SUMMARY] Overall Scraping Session Results:")
    logging.info(f"Session Duration: {summary['session_duration']['total_seconds']:.2f} seconds")
    
    # URLs processed
    logging.info("\n[URLS] Processing Statistics:")
    logging.info(f"• Total URLs processed: {summary['urls_processed']['total']}")
    logging.info(f"• Successful: {summary['urls_processed']['successful']}")
    logging.info(f"• Failed: {summary['urls_processed']['failed']}")
    
    # OCR metrics
    logging.info("\n[OCR] Overall OCR Statistics:")
    logging.info(f"• Total OCR attempts: {summary['ocr_metrics']['total_attempts']}")
    logging.info(f"• Total OCR successes: {summary['ocr_metrics']['total_successes']}")
    logging.info(f"• Average OCR success rate: {summary['ocr_metrics']['average_success_rate']:.2f}%")
    
    # Performance metrics
    logging.info("\n[PERFORMANCE] Processing Performance:")
    logging.info(f"• Total processing time: {summary['performance']['total_processing_time']:.2f} seconds")
    logging.info(f"• Average time per URL: {summary['performance']['average_time_per_url']:.2f} seconds")
    
    # Log failed URLs if any
    if session.failed_urls:
        logging.info("\n[FAILED] Failed URLs:")
        for url, error in session.failed_urls:
            logging.info(f"• {url}")
            if error is not None:
                logging.info(f"Error Type: {error.error_type}")
                logging.info(f"Error Message: {str(error)}")
                logging.info(f"Timestamp: {error.timestamp.isoformat()}")
                if error.details:
                    logging.info("Additional Details:")
                    for key, value in error.details.items():
                        logging.info(f"  {key}: {value}")
            else:
                logging.info("Error: Unknown error occurred")
    
    # Log JSON summary for machine parsing
    logging.info("\n[STATS] Session JSON Summary:")
    logging.info(json.dumps(summary, indent=2))

def read_urls_from_file(file_path: str) -> Iterator[str]:
    """Read URLs from a file, one per line.
    
    Args:
        file_path: Path to the file containing URLs
        
    Yields:
        str: Each URL from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        InvalidURLError: If the file is empty or contains invalid URLs
    """
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
            
        if not urls:
            raise InvalidURLError(f"No URLs found in file: {file_path}")
            
        for url in urls:
            yield url
            
    except FileNotFoundError:
        raise FileNotFoundError(f"URL file not found: {file_path}")

def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return level_map.get(level_str.upper(), logging.INFO)

def setup_logging(log_level: Optional[str] = None) -> None:
    """Configure logging to both file and console with timestamps and log levels.
    
    File logs will include emojis, while console logs will use text-based indicators.
    """
    # Create logs directory if it doesn't exist
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure file handler with emoji formatter
    file_handler = logging.FileHandler(str(config.LOG_FILE))
    file_handler.setFormatter(file_formatter)
    
    # Configure console handler with text-only formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Get log level from argument or environment variable
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(get_log_level(log_level))
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add a filter to the console handler to remove emojis
    class EmojiFilter(logging.Filter):
        def filter(self, record):
            # Replace emoji indicators with text equivalents
            record.msg = record.msg.replace('🟡', '[SUMMARY]')
            record.msg = record.msg.replace('📝', '[TEXT]')
            record.msg = record.msg.replace('🖼️', '[IMAGES]')
            record.msg = record.msg.replace('📚', '[FILE]')
            record.msg = record.msg.replace('✅', '[OK]')
            record.msg = record.msg.replace('❌', '[ERROR]')
            return True

    console_handler.addFilter(EmojiFilter())
    
    # Log the configured level
    logging.info(f"Logging configured with level: {log_level}")

def process_url(url: str, scrape_mode: str = 'both') -> None:
    """Process a single URL through the scraping pipeline.
    
    Args:
        url: The URL to scrape
        scrape_mode: What to scrape - 'text', 'ocr', or 'both'
        
    Raises:
        ScrapingError: If any error occurs during scraping
    """
    start_time = datetime.now()
    try:
        # Validate URL before attempting to scrape
        is_valid, error_message = validate_url(url)
        if not is_valid:
            raise InvalidURLError(error_message)
        logging.info(f"[OK] Valid URL format: {url}")
        
        # Log the start of scraping
        logging.info(f"[SCRAPE] Starting scrape for {url}")
        
        # Run the scraper
        result = scrape_page(url, scrape_mode)
        
        if not result:
            raise ParsingError("Failed to extract any content from the page")
        
        # Generate and log the summary
        summary = generate_scraping_summary(url, result, start_time)
        log_scraping_summary(summary)
            
    except InvalidURLError as e:
        logging.error(f"[ERROR] Invalid URL: {str(e)}")
        raise
    except ConnectionError as e:
        logging.error(f"[ERROR] Connection error: {str(e)}")
        raise
    except ParsingError as e:
        logging.error(f"[ERROR] Parsing error: {str(e)}")
        raise
    except RequestException as e:
        logging.error(f"[ERROR] Network error: {str(e)}")
        raise ConnectionError(f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        logging.error(f"[ERROR] Unexpected error: {str(e)}")
        raise ScrapingError(f"An unexpected error occurred: {str(e)}")

def write_session_log(session: ScrapingSession, run_dir: Path) -> None:
    """Write a detailed session log with comprehensive information about the scraping run.
    
    Args:
        session: The ScrapingSession object containing session metrics
        run_dir: The directory for this scraping run
    """
    # Generate log file path in the run directory
    log_file = run_dir / 'session.log'
    
    summary = session.get_session_summary()
    
    with open(log_file, 'w', encoding='utf-8') as f:
        # Write header with session information
        f.write("=" * 80 + "\n")
        f.write("SCRAPING SESSION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Session Information
        f.write("SESSION INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start Time: {summary['session_duration']['start']}\n")
        f.write(f"End Time: {summary['session_duration']['end']}\n")
        f.write(f"Total Duration: {summary['session_duration']['total_seconds']:.2f} seconds\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Platform: {sys.platform}\n")
        f.write(f"Run Directory: {run_dir}\n\n")
        
        # URL Processing Summary
        f.write("URL PROCESSING SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total URLs Processed: {summary['urls_processed']['total']}\n")
        f.write(f"Successful URLs: {summary['urls_processed']['successful']}\n")
        f.write(f"Failed URLs: {summary['urls_processed']['failed']}\n")
        success_rate = (summary['urls_processed']['successful'] / summary['urls_processed']['total'] * 100) if summary['urls_processed']['total'] > 0 else 0
        f.write(f"Success Rate: {success_rate:.1f}%\n\n")
        
        # Performance Metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Processing Time: {summary['performance']['total_processing_time']:.2f} seconds\n")
        f.write(f"Average Time per URL: {summary['performance']['average_time_per_url']:.2f} seconds\n")
        
        # Calculate processing speed safely
        if summary['performance']['total_processing_time'] > 0:
            processing_speed = summary['urls_processed']['total'] / (summary['performance']['total_processing_time'] / 60)
            f.write(f"Processing Speed: {processing_speed:.1f} URLs/minute\n")
        else:
            f.write("Processing Speed: N/A (no processing time recorded)\n")
        f.write("\n")
        
        # OCR Statistics
        f.write("OCR STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total OCR Attempts: {summary['ocr_metrics']['total_attempts']}\n")
        f.write(f"Total OCR Successes: {summary['ocr_metrics']['total_successes']}\n")
        f.write(f"Average OCR Success Rate: {summary['ocr_metrics']['average_success_rate']:.2f}%\n\n")
        
        # Successful URLs
        f.write("SUCCESSFUL URLS\n")
        f.write("-" * 80 + "\n")
        for url in session.successful_urls:
            f.write(f"• {url}\n")
        f.write("\n")
        
        # Failed URLs with Error Details
        if session.failed_urls:
            f.write("FAILED URLS AND ERRORS\n")
            f.write("-" * 80 + "\n")
            for url, error in session.failed_urls:
                f.write(f"URL: {url}\n")
                if error is not None:
                    f.write(f"Error Type: {error.error_type}\n")
                    f.write(f"Error Message: {str(error)}\n")
                    f.write(f"Timestamp: {error.timestamp.isoformat()}\n")
                    if error.details:
                        f.write("Additional Details:\n")
                        for key, value in error.details.items():
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write("Error: Unknown error occurred\n")
                f.write("\n")
        
        # System Information
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"CPU Count: {os.cpu_count()}\n")
        f.write(f"Memory Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB\n")
        f.write(f"Disk Space Available: {psutil.disk_usage('.').free / 1024 / 1024 / 1024:.1f} GB\n\n")
        
        # JSON Summary for Machine Parsing
        f.write("JSON SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(json.dumps(summary, indent=2))
    
    logging.info(f"\n[LOG] Session log written to: {log_file}")

def update_history_log(session: ScrapingSession) -> None:
    """Update the history log with a summary of the current scraping session.
    
    Args:
        session: The ScrapingSession object containing session metrics
    """
    # Use the same directory as the main log file
    log_dir = config.LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    history_file = log_dir / 'scrape_history.log'
    summary = session.get_session_summary()
    
    # Format the session summary for the history log
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_urls = summary['urls_processed']['total']
    avg_ocr_rate = summary['ocr_metrics']['average_success_rate']
    time_taken = summary['session_duration']['total_seconds']
    
    # Create the concise history entry
    history_entry = (
        f"{timestamp} | {total_urls} URLs Scraped | "
        f"Avg OCR Success Rate: {avg_ocr_rate:.1f}% | "
        f"Time Taken: {time_taken:.1f}s"
    )
    
    # Add error summary if there were any failures
    if session.failed_urls:
        error_types = {}
        for _, error in session.failed_urls:
            if error is not None:
                error_type = error.error_type if hasattr(error, 'error_type') else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1
            else:
                error_types['Unknown'] = error_types.get('Unknown', 0) + 1
        
        if error_types:
            error_summary = " | Errors: " + ", ".join(
                f"{error_type}({count})" for error_type, count in error_types.items()
            )
            history_entry += error_summary
    
    history_entry += "\n"
    
    try:
        # Append the entry to the history log
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(history_entry)
        
        # If we successfully wrote to the new location, try to migrate old logs
        old_log_dir = Path('scrape_logs')
        old_history_file = old_log_dir / 'scrape_history.log'
        if old_history_file.exists():
            try:
                # Read old logs
                with open(old_history_file, 'r', encoding='utf-8') as old_f:
                    old_entries = old_f.readlines()
                
                # Append old entries to new file
                with open(history_file, 'a', encoding='utf-8') as new_f:
                    new_f.writelines(old_entries)
                
                # Remove old file and directory if empty
                old_history_file.unlink()
                if not any(old_log_dir.iterdir()):
                    old_log_dir.rmdir()
                
                logging.info(f"Successfully migrated old history logs to {history_file}")
            except Exception as e:
                logging.warning(f"Failed to migrate old history logs: {str(e)}")
        
        logging.info(f"\n[LOG] Session summary added to history log: {history_file.absolute()}")
    except Exception as e:
        logging.error(f"Failed to write to history log: {str(e)}")

def write_run_summary(session: ScrapingSession, run_dir: Path) -> None:
    """Write a detailed summary of the scraping run to a JSON file.
    
    Args:
        session: The ScrapingSession object containing session metrics
        run_dir: The directory for this scraping run
    """
    summary = session.get_session_summary()
    
    # Add additional run-specific information
    run_summary = {
        'run_info': {
            'timestamp': datetime.now().isoformat(),
            'run_directory': str(run_dir),
            'command_line': ' '.join(sys.argv),
            'python_version': sys.version,
            'platform': sys.platform
        },
        'session_metrics': summary,
        'failed_urls': [
            {
                'url': url,
                'error_type': error.error_type if error else 'Unknown',
                'error_message': str(error) if error else 'Unknown error',
                'timestamp': error.timestamp.isoformat() if error else None,
                'details': error.details if error and error.details else None
            }
            for url, error in session.failed_urls
        ],
        'successful_urls': session.successful_urls,
        'performance': {
            'total_duration': summary['session_duration']['total_seconds'],
            'average_time_per_url': summary['performance']['average_time_per_url'],
            'start_time': summary['session_duration']['start'],
            'end_time': summary['session_duration']['end']
        },
        'content_metrics': {
            'total_urls_processed': summary['urls_processed']['total'],
            'successful_scrapes': summary['urls_processed']['successful'],
            'failed_scrapes': summary['urls_processed']['failed'],
            'success_rate': (summary['urls_processed']['successful'] / summary['urls_processed']['total'] * 100) if summary['urls_processed']['total'] > 0 else 0
        }
    }
    
    # Write the summary to a JSON file
    summary_file = run_dir / 'summary.json'
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)
        logging.info(f"\n[SUMMARY] Run summary written to: {summary_file}")
    except Exception as e:
        logging.error(f"Failed to write run summary: {str(e)}")

def main() -> None:
    parser = argparse.ArgumentParser(description='Scrape a webpage and extract text and images.')
    parser.add_argument('url_pos', nargs='?', help='URL to scrape')
    parser.add_argument('--url', help='URL to scrape (alternative to positional argument)')
    parser.add_argument('--url-file', help='Path to file containing URLs to scrape (one per line)')
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    parser.add_argument('--output-dir',
                       help='Custom directory where scraper results will be saved')
    parser.add_argument('--scrape-mode',
                       choices=['text', 'ocr', 'both'],
                       default='both',
                       help='What to scrape: text only, OCR only, or both (default: both)')
    parser.add_argument('--run-name',
                       help='Name for this scraping run (will be included in the run directory name)')
    args = parser.parse_args()

    # Initialize logging
    setup_logging(log_level=args.log_level)
    
    # Set custom output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        config.set_output_directory(output_dir)
    
    # Initialize a new run directory with optional name
    run_dir = config.initialize_run_directory(args.run_name)
    logging.info(f"Created new run directory: {run_dir}")
    
    # Initialize session tracking
    session = ScrapingSession()
    
    try:
        if args.url_file:
            # Process URLs from file
            logging.info(f"[TEXT] Processing URLs from file: {args.url_file}")
            try:
                for url in read_urls_from_file(args.url_file):
                    try:
                        start_time = datetime.now()
                        result = scrape_page(url, args.scrape_mode)
                        if result:
                            summary = generate_scraping_summary(url, result, start_time)
                            session.add_url_result(url, summary, True)
                        else:
                            session.add_url_result(url, {
                                'timestamp': {'duration_seconds': 0},
                                'extraction': {'metrics': {'ocr_attempts': 0, 'ocr_successes': 0}}
                            }, False)
                    except KeyboardInterrupt:
                        logging.info("\n[INTERRUPT] Scraping interrupted by user. Saving current progress...")
                        break
                    except ScrapingError as e:
                        logging.error(f"Failed to process URL: {url}")
                        logging.error(f"Error Type: {e.error_type}")
                        logging.error(f"Error Message: {str(e)}")
                        if e.details:
                            logging.error("Additional Details:")
                            for key, value in e.details.items():
                                logging.error(f"  {key}: {value}")
                        session.add_url_result(url, {
                            'timestamp': {'duration_seconds': 0},
                            'extraction': {'metrics': {'ocr_attempts': 0, 'ocr_successes': 0}}
                        }, False, e)
                        continue
            except KeyboardInterrupt:
                logging.info("\n[INTERRUPT] Scraping interrupted by user. Saving current progress...")
        else:
            # Process single URL
            url = args.url or args.url_pos
            if not url:
                parser.error("URL is required. Provide it either as a positional argument, --url, or --url-file")
            try:
                start_time = datetime.now()
                result = scrape_page(url, args.scrape_mode)
                if result:
                    summary = generate_scraping_summary(url, result, start_time)
                    session.add_url_result(url, summary, True)
                else:
                    session.add_url_result(url, {
                        'timestamp': {'duration_seconds': 0},
                        'extraction': {'metrics': {'ocr_attempts': 0, 'ocr_successes': 0}}
                    }, False)
            except KeyboardInterrupt:
                logging.info("\n[INTERRUPT] Scraping interrupted by user. Saving current progress...")
            except ScrapingError as e:
                logging.error(f"Failed to process URL: {url}")
                logging.error(f"Error Type: {e.error_type}")
                logging.error(f"Error Message: {str(e)}")
                if e.details:
                    logging.error("Additional Details:")
                    for key, value in e.details.items():
                        logging.error(f"  {key}: {value}")
                session.add_url_result(url, {
                    'timestamp': {'duration_seconds': 0},
                    'extraction': {'metrics': {'ocr_attempts': 0, 'ocr_successes': 0}}
                }, False, e)
        
        # Log session summary to console
        log_session_summary(session)
        
        # Write detailed session log in the run directory
        write_session_log(session, run_dir)
        
        # Update history log
        update_history_log(session)
        
        # Write run summary
        write_run_summary(session, run_dir)
            
    except KeyboardInterrupt:
        logging.info("\n[INTERRUPT] Scraping interrupted by user. Saving current progress...")
        # Save what we have so far
        log_session_summary(session)
        write_session_log(session, run_dir)
        update_history_log(session)
        write_run_summary(session, run_dir)
        logging.info("[DONE] Progress saved. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"[ERROR] Fatal error: {str(e)}")
        raise
    finally:
        # Ensure all Playwright browsers are closed
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                # Try to close any active browser contexts
                try:
                    # Get the default browser if it exists
                    browser = getattr(p, '_default_browser', None)
                    if browser and browser.is_connected():
                        browser.close()
                except Exception as e:
                    logging.debug(f"No active browser to close: {str(e)}")
        except Exception as e:
            logging.warning(f"Error during browser cleanup: {str(e)}")
        finally:
            # Force cleanup of any remaining Playwright processes
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if 'playwright' in proc.info['name'].lower():
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception as e:
                logging.warning(f"Error during process cleanup: {str(e)}")

if __name__ == "__main__":
    main() 