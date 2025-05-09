import os
import traceback
import sys
import json
import re
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright, TimeoutError, Error as PlaywrightError
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

from .utils import (
    download_image, 
    get_safe_filename, 
    create_metadata, 
    process_image_for_ocr, 
    validate_url,
    create_scraper_directories,
    normalize_hostname
)
from .ocr import ocr_image
from . import config

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def get_hostname(url: str) -> str:
    """Extract hostname from URL and convert to a safe filename."""
    return normalize_hostname(url)

def save_ocr_results(ocr_dir: Path, ocr_results: List[Dict[str, Any]], url: str, hostname: str) -> Path:
    """Save OCR results with metadata."""
    logging.info(f"Saving OCR results for {url} ({len(ocr_results)} images)")
    
    # Save individual OCR results
    for idx, result in enumerate(ocr_results):
        # Create a unique filename for each OCR result
        ocr_filename = f"ocr_{idx+1:03d}_{get_safe_filename(result['image_url'])}.json"
        ocr_path = ocr_dir / ocr_filename
        
        # Add metadata to the OCR result
        ocr_data = create_metadata(url, hostname)
        ocr_data.update({
            'image_url': result['image_url'],
            'image_path': str(result['image_path']),
            'ocr_text': result['ocr_text'],
            'ocr_text_length': result['ocr_text_length'],
            'ocr_text_word_count': result['ocr_text_word_count']
        })
        
        try:
            with open(ocr_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved OCR result {idx+1}/{len(ocr_results)} to {ocr_path}")
        except Exception as e:
            logging.error(f"Failed to save OCR result {idx+1} to {ocr_path}: {e}")
    
    # Save summary of all OCR results
    summary = create_metadata(url, hostname, ocr_results=ocr_results)
    
    summary_path = ocr_dir / 'summary.json'
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved OCR summary to {summary_path}")
        logging.info(f"Total OCR text length: {summary['total_ocr_text_length']} characters")
        logging.info(f"Total OCR word count: {summary['total_ocr_word_count']} words")
    except Exception as e:
        logging.error(f"Failed to save OCR summary to {summary_path}: {e}")
    
    return summary_path

def scrape_page(url: str, scrape_mode: str = 'both') -> Optional[Dict[str, Any]]:
    """Scrape a webpage and extract text, images, and perform OCR.
    
    Args:
        url: The URL to scrape
        scrape_mode: What to scrape - 'text', 'ocr', or 'both'
        
    Returns:
        Dict containing scraping results and metadata, or None if scraping fails
        
    Raises:
        ValueError: If URL is invalid
        ConnectionError: If connection fails
        TimeoutError: If request times out
        RuntimeError: If scraping fails
    """
    try:
        logging.info(f"[OK] Scraping {url} in mode: {scrape_mode}")
        
        # Validate URL format
        is_valid, error_message = validate_url(url)
        if not is_valid:
            logging.error(f"[ERROR] Invalid URL: {error_message}")
            raise ValueError(error_message)
        
        # Get hostname and run directory
        hostname = normalize_hostname(url)
        run_dir = config.get_run_directory()
        
        # Get output paths
        paths = {
            'base_dir': run_dir,
            'images_dir': run_dir / 'images' / hostname,
            'pages_dir': run_dir / 'pages' / hostname,
            'ocr_dir': run_dir / 'pages' / hostname / 'ocr'
        }
        
        # Create directories if they don't exist
        for dir_path in paths.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Created directory: {dir_path}")
        
        # Initialize metrics
        metrics = {
            'browser_init': 0.0,
            'page_load': 0.0,
            'content_extraction': 0.0,
            'image_processing': {
                'count': 0,
                'successful': 0,
                'failed': 0,
                'total': 0.0
            },
            'file_saving': 0.0,
            'total_time': 0.0
        }
        
        start_time = time.time()
        
        # Initialize browser and load page
        browser_init_start = time.time()
        with sync_playwright() as p:
            try:
                with p.chromium.launch(headless=True) as browser:
                    with browser.new_context() as context:
                        with context.new_page() as page:
                            page.set_default_timeout(30000)
                            
                            # Load page
                            page_load_start = time.time()
                            try:
                                response = page.goto(url)
                                if not response:
                                    raise RuntimeError(f"Failed to load {url}: No response received")
                                if not response.ok:
                                    raise RuntimeError(f"Failed to load {url}: HTTP {response.status} - {response.status_text}")
                            except TimeoutError:
                                raise RuntimeError(f"Timeout while loading {url} after 30 seconds")
                            except PlaywrightError as e:
                                raise RuntimeError(f"Playwright error while loading {url}: {str(e)}")

                            try:
                                page.wait_for_load_state('networkidle')
                            except TimeoutError:
                                logging.warning(f"Timeout waiting for network idle on {url}, continuing anyway")
                            except Exception as e:
                                logging.warning(f"Error waiting for network idle: {str(e)}, continuing anyway")
                            metrics['page_load'] = time.time() - page_load_start

                            # Extract content based on scrape mode
                            content_extraction_start = time.time()
                            html_content = None
                            visible_text = None
                            cleaned_text = None
                            ocr_results = []
                            failed_images = []

                            if scrape_mode in ['text', 'both']:
                                try:
                                    html_content = page.content()
                                    visible_text = page.inner_text('body')
                                    cleaned_text = clean_text(visible_text)
                                except Exception as e:
                                    raise RuntimeError(f"Failed to extract page content: {str(e)}")

                            if scrape_mode in ['ocr', 'both']:
                                try:
                                    # Process images
                                    image_processing_start = time.time()
                                    images = page.query_selector_all('img')
                                    metrics['image_processing']['count'] = len(images)
                                    
                                    for img in images:
                                        try:
                                            src = img.get_attribute('src')
                                            if not src:
                                                continue
                                                
                                            # Download and process image
                                            img_path = download_image(src, paths['images_dir'])
                                            if not img_path:
                                                continue
                                                
                                            # Perform OCR
                                            ocr_result = ocr_image(img_path)
                                            if ocr_result:
                                                ocr_results.append({
                                                    'image_path': str(img_path),
                                                    'text': ocr_result
                                                })
                                                metrics['image_processing']['successful'] += 1
                                            else:
                                                failed_images.append(str(img_path))
                                                metrics['image_processing']['failed'] += 1
                                                
                                        except Exception as e:
                                            logging.error(f"Failed to process image: {str(e)}")
                                            failed_images.append(str(img_path))
                                            metrics['image_processing']['failed'] += 1
                                            
                                    metrics['image_processing']['total'] = time.time() - image_processing_start
                                except Exception as e:
                                    raise RuntimeError(f"Failed to process images: {str(e)}")

                            metrics['content_extraction'] = time.time() - content_extraction_start

                            # Save files
                            file_saving_start = time.time()
                            try:
                                # Save page content
                                page_path = paths['pages_dir'] / "page.html"
                                with open(page_path, 'w', encoding='utf-8') as f:
                                    f.write(html_content)

                                # Save visible text with metadata
                                text_data = create_metadata(url, hostname, text=cleaned_text)
                                text_data['ocr_results_count'] = len(ocr_results)
                                text_data['images_dir'] = str(paths['images_dir'])
                                text_data['failed_images'] = failed_images

                                # Save as JSON
                                text_path = paths['pages_dir'] / "text.json"
                                with open(text_path, 'w', encoding='utf-8') as f:
                                    json.dump(text_data, f, indent=2, ensure_ascii=False)

                                # Save raw text
                                raw_text_path = paths['pages_dir'] / "text.txt"
                                with open(raw_text_path, 'w', encoding='utf-8') as f:
                                    f.write(cleaned_text)

                                # Save OCR results
                                ocr_summary_path = save_ocr_results(paths['ocr_dir'], ocr_results, url, hostname)
                            except Exception as e:
                                raise RuntimeError(f"Failed to save files: {str(e)}")
                            metrics['file_saving'] = time.time() - file_saving_start

                            # Calculate total time
                            metrics['total_time'] = time.time() - start_time

                            # Log performance metrics
                            logging.info("\n[STATS] Performance Metrics:")
                            logging.info(f"Total scraping time: {metrics['total_time']:.2f}s")
                            logging.info(f"Browser initialization: {metrics['browser_init']:.2f}s")
                            logging.info(f"Page loading: {metrics['page_load']:.2f}s")
                            logging.info(f"Content extraction: {metrics['content_extraction']:.2f}s")
                            logging.info(f"Image processing: {metrics['image_processing']['total']:.2f}s")
                            logging.info(f"  • Total images: {metrics['image_processing']['count']}")
                            logging.info(f"  • Successful: {metrics['image_processing']['successful']}")
                            logging.info(f"  • Failed: {metrics['image_processing']['failed']}")
                            logging.info(f"File saving: {metrics['file_saving']:.2f}s")

                            return {
                                'html': html_content,
                                'text': cleaned_text,
                                'text_data': text_data,
                                'images': ocr_results,
                                'failed_images': failed_images,
                                'page_path': page_path,
                                'text_path': text_path,
                                'raw_text_path': raw_text_path,
                                'ocr_dir': paths['ocr_dir'],
                                'ocr_summary_path': ocr_summary_path,
                                'images_dir': paths['images_dir'],
                                'metrics': metrics
                            }

            except Exception as e:
                raise RuntimeError(f"Failed to initialize browser: {str(e)}")
        metrics['browser_init'] = time.time() - browser_init_start

    except ValueError as e:
        logging.error(f"[ERROR] Invalid URL: {str(e)}")
        return None
    except RuntimeError as e:
        logging.error(f"[ERROR] Scraping error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"[ERROR] Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return None
