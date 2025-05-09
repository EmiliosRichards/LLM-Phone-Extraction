# Scraper Usage Guide

This document provides detailed information about the command-line arguments and usage examples for the web scraper.

## Command-Line Arguments

### Basic Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `url_pos` | URL to scrape (positional argument) | Yes* | - |
| `--url` | URL to scrape (alternative to positional argument) | Yes* | - |
| `--url-file` | Path to file containing URLs to scrape (one per line) | Yes* | - |

*Note: One of `url_pos`, `--url`, or `--url-file` must be provided.

### Optional Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--scrape-mode` | What to scrape | 'both' | 'text', 'ocr', 'both' |
| `--output-dir` | Custom directory for output files | './data' | - |
| `--log-level` | Set logging level | 'DEBUG' | 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' |
| `--run-name` | Name for this scraping run | - | - |

## Usage Examples

### Basic Usage

1. Scrape a single URL using positional argument:
```bash
python -m scraper.main "https://example.com"
```

2. Scrape a single URL using --url flag:
```bash
python -m scraper.main --url "https://example.com"
```

3. Scrape multiple URLs from a file:
```bash
python -m scraper.main --url-file urls.txt
```

### Scraping Modes

1. Scrape text content only:
```bash
python -m scraper.main --url "https://example.com" --scrape-mode text
```

2. Scrape images and perform OCR only:
```bash
python -m scraper.main --url "https://example.com" --scrape-mode ocr
```

3. Scrape both text and images with OCR (default):
```bash
python -m scraper.main --url "https://example.com" --scrape-mode both
```

### Output and Logging

1. Specify custom output directory:
```bash
python -m scraper.main --url "https://example.com" --output-dir "./custom_output"
```

2. Set logging level:
```bash
python -m scraper.main --url "https://example.com" --log-level INFO
```

3. Name your scraping run:
```bash
python -m scraper.main --url "https://example.com" --run-name "my_scrape_run"
```

### Combined Examples

1. Full featured example with all options:
```bash
python -m scraper.main --url "https://example.com" --scrape-mode both --output-dir "./data" --log-level INFO --run-name "test_run"
```

2. Process multiple URLs with custom settings:
```bash
python -m scraper.main --url-file urls.txt --scrape-mode both --output-dir "./data" --log-level DEBUG --run-name "batch_run"
```

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

## Environment Variables

The scraper can be configured using environment variables:

- `SCRAPER_ROOT`: Root directory for the project (default: current working directory)
- `SCRAPER_LOG_LEVEL`: Logging level (default: DEBUG)
- `SCRAPER_IMAGE_TIMEOUT`: Timeout for image downloads in seconds (default: 10)
- `SCRAPER_IMAGE_RETRY_COUNT`: Number of retry attempts for failed image downloads (default: 3)
- `SCRAPER_IMAGE_RETRY_DELAY`: Delay between image download retries in seconds (default: 1)
- `SCRAPER_MAX_REQUESTS_PER_SECOND`: Rate limit for requests (default: 2.0)
- `SCRAPER_RATE_LIMIT_BURST`: Maximum burst of requests allowed (default: 5) 