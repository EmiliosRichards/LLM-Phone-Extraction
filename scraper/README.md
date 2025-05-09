# Web Scraper Module

A robust web scraping module that extracts text content and images from web pages, with built-in OCR capabilities for image text extraction.

## Purpose

This web scraper is specifically designed to extract phone numbers and contact information from web pages. It combines powerful text extraction capabilities with OCR (Optical Character Recognition) to ensure comprehensive coverage of both visible text and text embedded within images.

### Key Objectives
- Extract phone numbers and contact information from web pages
- Process both visible text content and text within images
- Handle various phone number formats and representations
- Provide reliable extraction from contact pages, business listings, and other relevant content

### Use Cases
- Extracting phone numbers from business contact pages
- Gathering contact information from directory listings
- Processing business cards and contact images
- Collecting phone numbers from various web sources

## Features

- **Web Page Scraping**: Extracts text content and images from web pages using Playwright
- **Image Processing**: Downloads and processes images from web pages
- **OCR Integration**: Performs Optical Character Recognition on images to extract text
- **Rate Limiting**: Built-in rate limiting to prevent overwhelming target servers
- **Robust Error Handling**: Comprehensive error handling and logging
- **Configurable**: Highly configurable through environment variables
- **Detailed Logging**: Extensive logging with configurable log levels
- **Output Organization**: Structured output with organized file storage

## Directory Structure

```
scraper/
├── __init__.py
├── config.py          # Configuration settings and constants
├── main.py           # Main entry point and CLI interface
├── ocr.py            # OCR functionality for image text extraction
├── rate_limiter.py   # Rate limiting implementation
├── scraper.py        # Core scraping functionality
└── utils.py          # Utility functions
```

## Configuration

The scraper can be configured using environment variables:

- `SCRAPER_ROOT`: Root directory for the project (default: current working directory)
- `SCRAPER_LOG_LEVEL`: Logging level (default: DEBUG)
- `SCRAPER_IMAGE_TIMEOUT`: Timeout for image downloads in seconds (default: 10)
- `SCRAPER_IMAGE_RETRY_COUNT`: Number of retry attempts for failed image downloads (default: 3)
- `SCRAPER_IMAGE_RETRY_DELAY`: Delay between image download retries in seconds (default: 1)
- `SCRAPER_MAX_REQUESTS_PER_SECOND`: Rate limit for requests (default: 2.0)
- `SCRAPER_RATE_LIMIT_BURST`: Maximum burst of requests allowed (default: 5)

## Output Structure

The scraper organizes its output in the following structure:

```
data/
├── images/           # Downloaded images
│   └── {hostname}/   # Images organized by hostname
├── pages/           # Scraped page content
│   └── {hostname}/   # Pages organized by hostname
│       └── ocr/     # OCR results for images
└── logs/            # Log files
```

## Usage

### Basic Usage

```python
from scraper import scrape_page

# Scrape a single URL
result = scrape_page("https://example.com")

# Handle common errors
try:
    result = scrape_page("https://example.com")
except NetworkError:
    print("Failed to connect to the website. Please check your internet connection.")
except InvalidURLError:
    print("The provided URL is invalid. Please check the URL format.")
except TimeoutError:
    print("The request timed out. The website might be slow or unresponsive.")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
```

### Command Line Interface

The scraper provides a powerful command-line interface with various options:

```bash
# Basic usage with a single URL (as positional argument)
python -m scraper.main "https://example.com"

# Alternative way to specify URL
python -m scraper.main --url "https://example.com"

# Scrape multiple URLs from a file
python -m scraper.main --url-file urls.txt

# Specify custom output directory
python -m scraper.main --url "https://example.com" --output-dir "./custom_output"

# Set logging level
python -m scraper.main --url "https://example.com" --log-level INFO

# Combine multiple options
python -m scraper.main --url-file urls.txt --output-dir "./data" --log-level DEBUG
```

#### Available CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `url_pos` | URL to scrape (positional argument) | Required if no --url |
| `--url` | URL to scrape (alternative to positional argument) | Required if no positional argument |
| `--url-file` | Path to file containing URLs to scrape (one per line) | None |
| `--output-dir` | Custom directory for output files | `./data` |
| `--log-level` | Set logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) | DEBUG |

### Error Handling

The scraper implements robust error handling for common scenarios:

1. **Network Errors**
   - Connection timeouts
   - DNS resolution failures
   - SSL/TLS errors
   - Server errors (5xx)

2. **Content Errors**
   - Invalid HTML
   - Missing content
   - Encoding issues
   - Malformed images

3. **Resource Errors**
   - Disk space issues
   - Permission problems
   - File system errors

4. **Rate Limiting**
   - Too many requests
   - IP blocking
   - Server throttling

Each error type is logged with appropriate context and severity level, allowing for proper debugging and monitoring.

#### Common Error Scenarios and Handling

```python
from scraper import ScraperError, NetworkError, ContentError, RateLimitError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scraper')

try:
    # Example: Handling network timeout
    result = scraper.scrape_page("https://example.com")
except NetworkError as e:
    logger.error(f"Network error occurred: {str(e)}")
    if e.is_timeout:
        logger.info("Attempting retry with increased timeout...")
        result = scraper.scrape_page("https://example.com", timeout=60)
    elif e.is_dns_error:
        logger.warning("DNS resolution failed, checking alternative domain...")
        result = scraper.scrape_page("https://www.example.com")
    else:
        raise

# Example: Handling content errors
try:
    content = scraper.extract_content(html)
except ContentError as e:
    logger.error(f"Content extraction failed: {str(e)}")
    if e.is_encoding_error:
        logger.info("Attempting with different encoding...")
        content = scraper.extract_content(html, encoding='utf-8')
    elif e.is_malformed_html:
        logger.warning("HTML is malformed, attempting cleanup...")
        content = scraper.extract_content(scraper.clean_html(html))
    else:
        raise

# Example: Handling rate limiting
try:
    results = scraper.scrape_batch(urls)
except RateLimitError as e:
    logger.warning(f"Rate limit exceeded: {str(e)}")
    if e.retry_after:
        logger.info(f"Waiting {e.retry_after} seconds before retry...")
        time.sleep(e.retry_after)
        results = scraper.scrape_batch(urls)
    else:
        # Implement exponential backoff
        wait_time = min(300, 2 ** e.attempt * 5)
        logger.info(f"Implementing exponential backoff: {wait_time}s")
        time.sleep(wait_time)
        results = scraper.scrape_batch(urls)
```

#### Error Logging and Monitoring

The scraper provides detailed logging for each error type:

```python
# Example log output for different error types
2024-03-14 10:15:23,456 - scraper - ERROR - Network error: Connection timeout after 30s
2024-03-14 10:15:24,789 - scraper - WARNING - Rate limit exceeded: 429 Too Many Requests
2024-03-14 10:15:25,123 - scraper - INFO - Retrying with exponential backoff: 5s
2024-03-14 10:15:30,456 - scraper - ERROR - Content error: Invalid HTML structure
```

#### Error Recovery Strategies

1. **Network Errors**
   ```python
   # Automatic retry with exponential backoff
   config = {
       'max_retries': 3,
       'initial_delay': 1,
       'max_delay': 30,
       'backoff_factor': 2
   }
   
   try:
       result = scraper.scrape_with_retry(url, config)
   except NetworkError as e:
       logger.error(f"Failed after {config['max_retries']} retries: {str(e)}")
   ```

2. **Content Errors**
   ```python
   # Fallback content extraction
   try:
       content = scraper.extract_content(html)
   except ContentError:
       # Try alternative extraction methods
       content = scraper.extract_content_fallback(html)
       logger.info("Using fallback content extraction method")
   ```

3. **Rate Limiting**
   ```python
   # Dynamic rate limit adjustment
   try:
       results = scraper.scrape_batch(urls)
   except RateLimitError as e:
       # Adjust rate limit based on server response
       new_rate_limit = e.suggested_rate_limit
       scraper.configure_rate_limit(new_rate_limit)
       logger.info(f"Adjusted rate limit to {new_rate_limit} requests/second")
       results = scraper.scrape_batch(urls)
   ```

#### Error Monitoring and Alerts

The scraper can be configured to send alerts for critical errors:

```python
from scraper import ErrorMonitor

# Configure error monitoring
monitor = ErrorMonitor(
    alert_threshold=5,  # Number of errors before alert
    alert_interval=300,  # Alert interval in seconds
    alert_channels=['email', 'slack']
)

# Example alert configuration
monitor.configure_alerts({
    'email': {
        'recipients': ['admin@example.com'],
        'subject': 'Scraper Error Alert'
    },
    'slack': {
        'webhook_url': 'https://hooks.slack.com/services/...',
        'channel': '#scraper-alerts'
    }
})

# Monitor errors
try:
    result = scraper.scrape_page(url)
except Exception as e:
    monitor.log_error(e)
    if monitor.should_alert():
        monitor.send_alert(f"Critical error occurred: {str(e)}")
```

## Features in Detail

### Text Extraction
- Extracts visible text content from web pages
- Cleans and normalizes text
- Preserves paragraph structure
- Handles various text formats

### Image Processing
- Downloads images from web pages
- Supports multiple image formats
- Handles relative and absolute URLs
- Implements retry logic for failed downloads

### OCR Capabilities
- Performs OCR on downloaded images
- Extracts text from images
- Generates OCR summaries
- Supports multiple image formats

### Rate Limiting
- Implements token bucket algorithm
- Configurable request rate limits
- Burst handling for temporary spikes
- Prevents server overload

## Error Handling

The scraper implements comprehensive error handling for various scenarios:
- Invalid URLs
- Connection failures
- Timeout handling
- Image processing errors
- OCR failures

## Logging

The scraper provides detailed logging with:
- Configurable log levels
- Structured log output
- Performance metrics
- Error tracking
- Operation summaries

### Logging Levels

The scraper supports the following logging levels, from most to least verbose:

1. **DEBUG (10)**
   - Detailed information for debugging purposes
   - Raw HTML content before processing
   - Image processing steps and parameters
   - OCR processing details
   - Rate limiter token bucket state
   - Memory usage statistics
   - Example: `DEBUG - Processing image with dimensions 800x600, format: PNG`

2. **INFO (20)**
   - General operational information
   - Page scraping start/completion
   - Image download status
   - OCR processing start/completion
   - Rate limit adjustments
   - Example: `INFO - Successfully scraped page: https://example.com`

3. **WARNING (30)**
   - Non-critical issues that don't stop execution
   - Rate limit approaching threshold
   - Image download retries
   - OCR confidence below threshold
   - Memory usage approaching limit
   - Example: `WARNING - Rate limit approaching threshold: 80% of limit reached`

4. **ERROR (40)**
   - Serious issues that affect functionality
   - Failed page loads
   - Image processing failures
   - OCR processing errors
   - Rate limit exceeded
   - Example: `ERROR - Failed to download image: Connection timeout`

5. **CRITICAL (50)**
   - System-level failures
   - Out of memory conditions
   - Disk space exhaustion
   - Fatal OCR engine errors
   - Example: `CRITICAL - Insufficient disk space for image storage`

### Log Format

The default log format includes:
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

Example output:
```
2024-03-14 10:15:23,456 - scraper.page - INFO - Starting page scrape: https://example.com
2024-03-14 10:15:23,789 - scraper.image - DEBUG - Processing image: banner.png (800x600)
2024-03-14 10:15:24,123 - scraper.ocr - WARNING - Low OCR confidence: 65%
2024-03-14 10:15:24,456 - scraper.rate_limiter - ERROR - Rate limit exceeded
```

### Configuring Logging

You can configure logging through environment variables or programmatically:

```python
# Using environment variables
SCRAPER_LOG_LEVEL=DEBUG
SCRAPER_LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SCRAPER_LOG_FILE="scraper.log"

# Programmatic configuration
import logging
from scraper import configure_logging

configure_logging(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file="scraper.log"
)
```

### Log Categories

The scraper organizes logs into the following categories:

1. **Page Scraping**
   - URL processing
   - Content extraction
   - HTML parsing
   - Link discovery

2. **Image Processing**
   - Download status
   - Format conversion
   - Size optimization
   - Storage operations

3. **OCR Operations**
   - Engine initialization
   - Processing steps
   - Confidence scores
   - Text extraction

4. **Rate Limiting**
   - Request tracking
   - Token bucket state
   - Rate adjustments
   - Burst handling

5. **System Operations**
   - Memory usage
   - Disk space
   - Performance metrics
   - Resource allocation

### Log Rotation

The scraper supports log rotation to manage log file sizes:

```python
from scraper import configure_log_rotation

configure_log_rotation(
    max_bytes=10485760,  # 10MB
    backup_count=5,
    rotation_interval='midnight'
)
```

## Dependencies

### Core Dependencies
- Playwright: For web page rendering and content extraction
- Python standard library
- Additional requirements specified in requirements.txt

### OCR Dependencies
The scraper uses the following OCR libraries for image text extraction:

1. **Tesseract OCR (pytesseract)**
   - Primary OCR engine for text extraction from images
   - Version: 4.1.1 or higher
   - Language support: English (default), configurable for other languages

2. **OpenCV (opencv-python)**
   - Image preprocessing for improved OCR accuracy
   - Version: 4.5.0 or higher
   - Used for image enhancement and noise reduction

3. **Pillow (PIL)**
   - Image processing and format conversion
   - Version: 8.0.0 or higher
   - Required for image manipulation before OCR

### OCR Installation Guide

#### Windows Installation
1. Install Tesseract OCR:
   ```bash
   # Using Chocolatey
   choco install tesseract
   
   # Or download installer from:
   # https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. Install Python dependencies:
   ```bash
   pip install pytesseract opencv-python pillow
   ```

3. Configure Tesseract path:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

#### Linux Installation
1. Install Tesseract OCR:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install tesseract-ocr
   
   # Fedora
   sudo dnf install tesseract
   ```

2. Install Python dependencies:
   ```bash
   pip install pytesseract opencv-python pillow
   ```

#### macOS Installation
1. Install Tesseract OCR:
   ```bash
   # Using Homebrew
   brew install tesseract
   ```

2. Install Python dependencies:
   ```bash
   pip install pytesseract opencv-python pillow
   ```

### OCR Configuration

The scraper provides several configuration options for OCR:

```python
from scraper import configure_ocr

# Basic OCR configuration
configure_ocr({
    'tesseract_cmd': '/usr/bin/tesseract',  # Path to tesseract executable
    'lang': 'eng',                          # Language(s) for OCR
    'config': '--psm 3',                    # Page segmentation mode
    'dpi': 300,                             # Image DPI for processing
    'timeout': 30                           # OCR processing timeout
})

# Advanced OCR configuration with image preprocessing
configure_ocr({
    'preprocessing': {
        'resize': True,                     # Resize images for better accuracy
        'denoise': True,                    # Apply noise reduction
        'threshold': True,                  # Apply adaptive thresholding
        'deskew': True                      # Correct image orientation
    },
    'postprocessing': {
        'clean_text': True,                 # Clean extracted text
        'remove_noise': True,               # Remove OCR artifacts
        'normalize_spaces': True            # Normalize whitespace
    }
})
```

### OCR Performance Optimization

1. **Image Preprocessing**
   - Resize images to optimal dimensions
   - Convert to grayscale for better accuracy
   - Apply adaptive thresholding
   - Remove noise and artifacts

2. **Memory Management**
   - Process images in batches
   - Clear memory after processing
   - Use appropriate image formats

3. **Error Handling**
   - Handle OCR failures gracefully
   - Implement retry logic for failed attempts
   - Log OCR performance metrics

### Additional Language Support

To add support for additional languages:

1. Install language data:
   ```bash
   # Windows
   # Download language data from Tesseract GitHub repository
   
   # Linux
   sudo apt-get install tesseract-ocr-[lang]
   
   # macOS
   brew install tesseract-lang
   ```

2. Configure language in scraper:
   ```python
   configure_ocr({
       'lang': 'eng+fra+deu',  # Multiple languages
       'lang_data_path': '/path/to/lang/data'
   })
   ```

## Advanced Features

### Rate Limiting Configuration

The scraper implements a sophisticated token bucket algorithm for rate limiting. You can fine-tune the behavior through environment variables or configuration files:

```python
# Example configuration
SCRAPER_MAX_REQUESTS_PER_SECOND = 2.0  # Base rate limit
SCRAPER_RATE_LIMIT_BURST = 5          # Maximum burst size
SCRAPER_RATE_LIMIT_WINDOW = 60        # Time window in seconds
```

For different domains, you can set custom rate limits:

```python
from scraper import configure_rate_limits

# Set custom rate limits for specific domains
configure_rate_limits({
    "example.com": {
        "requests_per_second": 1.0,
        "burst_size": 3,
        "window_size": 30
    },
    "api.example.org": {
        "requests_per_second": 5.0,
        "burst_size": 10,
        "window_size": 60
    }
})
```

### Custom Retry Logic

The scraper allows customization of retry behavior for failed image downloads:

```python
from scraper import configure_retry_policy

# Configure custom retry policy
configure_retry_policy({
    "max_retries": 5,                 # Maximum number of retry attempts
    "initial_delay": 1,               # Initial delay in seconds
    "max_delay": 30,                  # Maximum delay between retries
    "backoff_factor": 2,              # Exponential backoff multiplier
    "retry_on_status": [408, 429, 500, 502, 503, 504]  # HTTP status codes to retry
})

# Custom retry handler for specific scenarios
def custom_retry_handler(attempt, error):
    if isinstance(error, NetworkError):
        return attempt < 3  # Retry up to 3 times for network errors
    return False  # Don't retry other errors
```

### Large Scale Scraping

For handling multiple URLs efficiently, the scraper supports concurrent processing:

```python
from scraper import scrape_batch
import asyncio

# Configure concurrent scraping
async def scrape_multiple_urls(urls):
    # Configure concurrency settings
    config = {
        "max_concurrent": 10,         # Maximum concurrent requests
        "batch_size": 100,            # URLs per batch
        "timeout": 300,               # Overall timeout in seconds
        "progress_callback": print    # Progress tracking
    }
    
    # Process URLs in batches
    results = await scrape_batch(urls, config)
    return results

# Example usage
urls = ["https://example.com/page1", "https://example.com/page2", ...]
results = asyncio.run(scrape_multiple_urls(urls))
```

#### Performance Optimization Tips

1. **Resource Management**
   - Adjust `max_concurrent` based on available system resources
   - Monitor memory usage during large batch operations
   - Use appropriate batch sizes to prevent overwhelming target servers

2. **Error Recovery**
   - Implement checkpointing for long-running operations
   - Save progress regularly to handle interruptions
   - Use exponential backoff for retry attempts

3. **Monitoring and Logging**
   - Enable detailed logging for debugging
   - Monitor rate limit compliance
   - Track success/failure rates

4. **Best Practices**
   - Respect robots.txt directives
   - Implement proper user agent rotation
   - Use appropriate delays between requests
   - Handle rate limiting responses gracefully

## Testing

The scraper includes a comprehensive test suite to ensure reliability and maintainability.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Run unit tests only
pytest tests/integration/   # Run integration tests only
pytest tests/e2e/          # Run end-to-end tests only

# Run tests with coverage report
pytest --cov=scraper --cov-report=html

# Run tests in parallel
pytest -n auto

# Run tests with specific markers
pytest -m "not slow"       # Skip slow tests
pytest -m "ocr"           # Run only OCR-related tests
```

### Test Categories

1. **Unit Tests**
   - Text extraction functions
   - URL parsing and validation
   - Rate limiting logic
   - Configuration handling
   - Error handling utilities

2. **Integration Tests**
   - Web page scraping
   - Image download and processing
   - OCR functionality
   - File system operations
   - Rate limiting with actual delays

3. **End-to-End Tests**
   - Complete scraping workflows
   - Multi-page scraping
   - Error recovery scenarios
   - Performance benchmarks

### Mocking External Services

#### Web Request Mocking

```python
import pytest
from unittest.mock import patch
from scraper import scrape_page

# Mock HTTP responses
@pytest.fixture
def mock_response():
    return {
        'status_code': 200,
        'text': '<html><body>Test content</body></html>',
        'headers': {'Content-Type': 'text/html'}
    }

def test_scrape_page(mock_response):
    with patch('requests.get', return_value=mock_response):
        result = scrape_page('https://example.com')
        assert result['content'] == 'Test content'

# Mock rate limiting
@pytest.fixture
def mock_rate_limiter():
    with patch('scraper.rate_limiter.RateLimiter') as mock:
        mock.return_value.acquire.return_value = True
        yield mock
```

#### OCR Mocking

```python
import pytest
from unittest.mock import patch
from scraper import extract_text_from_image

# Mock OCR results
@pytest.fixture
def mock_ocr():
    with patch('pytesseract.image_to_string') as mock:
        mock.return_value = "Mocked OCR text"
        yield mock

def test_image_ocr(mock_ocr):
    result = extract_text_from_image('test_image.png')
    assert result == "Mocked OCR text"
    mock_ocr.assert_called_once()

# Mock image processing
@pytest.fixture
def mock_image_processing():
    with patch('cv2.imread') as mock_read, \
         patch('cv2.cvtColor') as mock_convert, \
         patch('cv2.threshold') as mock_threshold:
        mock_read.return_value = np.zeros((100, 100))
        mock_convert.return_value = np.zeros((100, 100))
        mock_threshold.return_value = (0, np.zeros((100, 100)))
        yield {
            'read': mock_read,
            'convert': mock_convert,
            'threshold': mock_threshold
        }
```

### Test Data Management

```python
# Example test data structure
tests/
├── data/
│   ├── sample_pages/     # HTML test pages
│   ├── sample_images/    # Test images for OCR
│   └── expected_output/  # Expected results
├── fixtures/
│   ├── test_configs.py   # Test configurations
│   └── test_data.py      # Test data generators
└── mocks/
    ├── web_responses.py  # Mock web responses
    └── ocr_results.py    # Mock OCR results
```

### Writing Tests

1. **Unit Test Example**
   ```python
   def test_url_validation():
       assert is_valid_url("https://example.com")
       assert not is_valid_url("invalid-url")
       assert not is_valid_url("ftp://example.com")
   ```

2. **Integration Test Example**
   ```python
   @pytest.mark.integration
   def test_full_scraping_workflow():
       # Setup test environment
       test_url = "https://example.com"
       test_config = {
           'timeout': 5,
           'retry_count': 2
       }
       
       # Execute scraping
       result = scrape_page(test_url, config=test_config)
       
       # Verify results
       assert result['status'] == 'success'
       assert 'content' in result
       assert 'images' in result
   ```

3. **Performance Test Example**
   ```python
   @pytest.mark.performance
   def test_scraping_performance():
       urls = [f"https://example.com/page{i}" for i in range(10)]
       
       start_time = time.time()
       results = scrape_batch(urls)
       duration = time.time() - start_time
       
       assert duration < 30  # Should complete within 30 seconds
       assert len(results) == len(urls)
   ```

### Continuous Integration

The project includes GitHub Actions workflows for automated testing:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest --cov=scraper --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Contributing

When contributing to this module:
1. Follow the existing code style
2. Add appropriate error handling
3. Include logging for new features
4. Update documentation
5. Add tests for new functionality

## License

[Specify your license here] 