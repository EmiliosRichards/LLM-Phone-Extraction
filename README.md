# LLM Phone Number Extraction Pipeline

A robust pipeline for extracting phone numbers from text using Large Language Models (LLMs). The project includes two versions with different levels of sophistication and features.

## Purpose

This pipeline serves as a comprehensive solution for phone number extraction with the following key capabilities:

- **Multi-Source Extraction**: Extracts phone numbers from various data sources including text files and OCR results
- **Versioned Implementation**: Provides two distinct versions (v1 and v2) with different levels of sophistication
- **Structured Output**: Delivers well-formatted results including:
  - Extracted phone numbers with confidence scores
  - Categorization of numbers (e.g., Sales, Support)
  - Rich metadata and context preservation
  - Schema-validated output formats

## Overview

This pipeline uses LLMs to extract phone numbers from text, with support for:
- Multiple input sources (text files, OCR results)
- Structured output with confidence scores
- Metadata tracking
- Schema validation
- Batch processing

## How It Works

The pipeline operates in three main stages:

### 1. Data Collection
- **Text Input**: Processes raw text files and web content
- **OCR Processing**: 
  - Converts images and PDFs to text using Tesseract OCR
  - Handles multiple languages and formats
  - Pre-processes images for optimal text recognition
- **Batch Handling**: 
  - Supports parallel processing of multiple files
  - Maintains source tracking and metadata

### 2. LLM Processing
- **Text Analysis**:
  - Breaks input into manageable chunks
  - Preserves context and formatting
  - Handles various phone number formats
- **Number Extraction**:
  - Uses LLM to identify and validate phone numbers
  - Categorizes numbers based on context
  - Assigns confidence scores to each extraction
- **Context Preservation**:
  - Maintains surrounding text for verification
  - Tracks source and position information
  - Preserves formatting and structure

### 3. Output Generation
- **Structured Format**:
  - Generates JSON output with standardized schema
  - Includes metadata and processing information
  - Provides confidence scores and categorization
- **Validation**:
  - Validates phone number formats
  - Ensures schema compliance
  - Performs quality checks on output
- **Error Handling**:
  - Logs processing errors and warnings
  - Provides fallback mechanisms
  - Maintains processing statistics

## Project Structure

```
llm_pipeline/
├── common/                 # Shared utilities and base classes
│   ├── llm_base.py        # Base LLM client class
│   ├── schema_utils.py    # Schema validation utilities
│   ├── io_utils.py        # File I/O utilities
│   ├── text_utils.py      # Text processing utilities
│   └── log.py            # Logging utilities
├── v1/                    # Version 1 (Basic)
│   ├── llm_client.py     # Simple LLM client
│   ├── main.py           # Main processing script
│   └── test_prompt.py    # Prompt testing utility
├── v2/                    # Version 2 (Enhanced)
│   ├── llm_client.py     # Structured LLM client
│   ├── main.py           # Main processing script
│   └── test_prompt.py    # Prompt testing utility
├── config.py             # Shared configuration
└── prompts/              # Prompt templates
    ├── phone_extraction_prompt.txt
    └── phone_extraction_prompt_v2.txt
```

## Features

### Version 1 (Basic)
- Simple phone number extraction
- Basic categorization
- Raw text fallback
- Batch processing support

### Version 2 (Enhanced)
- Structured output format
- Confidence scores
- Context preservation
- Metadata tracking
- Schema validation
- Enhanced error handling
- Multiple data source support

## Installation

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space
- Internet connection for API access

### Setup Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-phone-extraction
```

2. Set up a virtual environment:

Using `venv` (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

Using `conda`:
```bash
# Create conda environment
conda create -n llm-pipeline python=3.8
conda activate llm-pipeline
```

3. Install dependencies:
```bash
# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies (for OCR support)
pip install -r requirements-ocr.txt
```

4. Configure the LLM API:
- Update `config.py` with your LLM API settings
- Set the appropriate model name
- Configure API endpoint

### Optional Setup

#### OCR Support
If you plan to use OCR features:
```bash
# Install Tesseract OCR
# On Windows:
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki

# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# On MacOS:
brew install tesseract
```

#### GPU Acceleration (Optional)
For improved performance with GPU support:
```bash
# Install CUDA toolkit if using NVIDIA GPU
# Then install GPU-enabled dependencies:
pip install -r requirements-gpu.txt
```

## Usage

### Basic Usage (v1)

```python
from llm_pipeline.v1.llm_client import V1LLMClient

# Create client instance
client = V1LLMClient()

# Process text
result = client.generate("""
    Contact our sales team at +1 (800) 555-1234 for pricing information.
    For technical support, please call +1 (888) 555-5678.
""")
```

### Enhanced Usage (v2)

```python
from llm_pipeline.v2.llm_client import V2LLMClient

# Create client instance
client = V2LLMClient()

# Process text
result = client.generate("""
    Contact our sales team at +1 (800) 555-1234 for pricing information.
    For technical support, please call +1 (888) 555-5678.
""")
```

### Command Line Usage

Process files using v1:
```bash
python -m llm_pipeline.v1.main --source text
```

Process files using v2:
```bash
python -m llm_pipeline.v2.main --source ocr+text --limit 50
```

Test prompts:
```bash
# Test v1 prompt
python -m llm_pipeline.v1.test_prompt

# Test v2 prompt
python -m llm_pipeline.v2.test_prompt --input-file path/to/text.txt
```

## Configuration

The pipeline uses a central `config.py` file for configuration. Here's a detailed breakdown of the configuration options:

### Basic Configuration

```python
# config.py

# API Settings
LLM_API_URL = "https://api.openai.com/v1"  # Your LLM provider's API endpoint
DEFAULT_MODEL_NAME = "gpt-4"  # or "gpt-3.5-turbo" for v1
API_KEY = os.getenv("LLM_API_KEY")  # Load from environment variable

# Directory Paths
BASE_DIR = Path(__file__).parent
PAGES_DIR = BASE_DIR / "data" / "pages"
OCR_DIR = BASE_DIR / "data" / "ocr"
OUTPUTS_V1 = BASE_DIR / "outputs" / "v1"
OUTPUTS_V2 = BASE_DIR / "outputs" / "v2"

# Prompt Paths
PROMPT_V1_PATH = BASE_DIR / "prompts" / "phone_extraction_prompt.txt"
PROMPT_V2_PATH = BASE_DIR / "prompts" / "phone_extraction_prompt_v2.txt"
```

### Environment-Specific Configuration

Create separate configuration files for different environments:

```python
# config_dev.py
DEBUG = True
LOG_LEVEL = "DEBUG"
BATCH_SIZE = 10
API_RATE_LIMIT = 100  # requests per minute

# config_prod.py
DEBUG = False
LOG_LEVEL = "INFO"
BATCH_SIZE = 50
API_RATE_LIMIT = 1000  # requests per minute
```

Load the appropriate configuration:
```python
import os

ENV = os.getenv("ENVIRONMENT", "dev")
if ENV == "prod":
    from config_prod import *
else:
    from config_dev import *
```

### Advanced Configuration Options

```python
# Processing Settings
CHUNK_SIZE = 1000  # Characters per chunk for processing
MAX_RETRIES = 3    # Number of API retry attempts
TIMEOUT = 30       # API request timeout in seconds

# Output Settings
OUTPUT_FORMAT = "json"  # or "csv"
INCLUDE_METADATA = True
SAVE_INTERMEDIATE = False

# OCR Settings
OCR_LANGUAGES = ["eng", "fra", "deu"]  # Supported languages
OCR_DPI = 300
OCR_TIMEOUT = 60
```

### Common Configuration Issues and Solutions

1. **API Connection Issues**
   - Problem: API requests failing
   - Solution: Check API key and endpoint URL
   ```python
   # Verify API connection
   import requests
   response = requests.get(LLM_API_URL, headers={"Authorization": f"Bearer {API_KEY}"})
   ```

2. **Directory Permission Errors**
   - Problem: Cannot write to output directories
   - Solution: Ensure proper permissions
   ```python
   # Add to config.py
   def ensure_directories():
       for directory in [PAGES_DIR, OCR_DIR, OUTPUTS_V1, OUTPUTS_V2]:
           directory.mkdir(parents=True, exist_ok=True)
   ```

3. **Memory Issues**
   - Problem: Out of memory during batch processing
   - Solution: Adjust batch size and chunk size
   ```python
   # Reduce memory usage
   BATCH_SIZE = 10  # Smaller batches
   CHUNK_SIZE = 500  # Smaller chunks
   ```

4. **OCR Performance**
   - Problem: Slow OCR processing
   - Solution: Optimize OCR settings
   ```python
   # Optimize OCR
   OCR_DPI = 200  # Lower DPI for faster processing
   OCR_LANGUAGES = ["eng"]  # Limit to required languages
   ```

### Configuration Best Practices

1. **Security**
   - Never commit API keys to version control
   - Use environment variables for sensitive data
   - Implement rate limiting for API calls

2. **Performance**
   - Adjust batch sizes based on available memory
   - Configure appropriate timeouts
   - Enable caching where possible

3. **Maintenance**
   - Document all configuration options
   - Use type hints for configuration values
   - Implement configuration validation

## Development

### Code Style and Standards

1. **Python Style Guide**
   - Follow PEP 8 guidelines
   - Use type hints for all function parameters and return values
   - Maximum line length: 88 characters (Black formatter)
   - Use meaningful variable and function names

2. **Code Formatting**
   ```bash
   # Install development dependencies
   pip install -r requirements-dev.txt

   # Format code
   black .
   isort .
   flake8 .

   # Type checking
   mypy .
   ```

3. **Documentation**
   - Use Google-style docstrings
   - Include type hints in docstrings
   - Document all public functions and classes
   ```python
   def process_text(text: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
       """Process input text and extract phone numbers.

       Args:
           text: Input text to process
           chunk_size: Size of text chunks for processing

       Returns:
           List of dictionaries containing extracted phone numbers and metadata
       """
   ```

### Feature Development Workflow

1. **Setting Up Development Environment**
   ```bash
   # Create feature branch
   git checkout -b feature/your-feature-name

   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

2. **Adding New Features**
   - **Common Utilities**:
     1. Add new module to `common/` directory
     2. Create corresponding test file
     3. Update `common/README.md`
     4. Add type hints and documentation
     5. Run tests and linting

   - **Version-Specific Features**:
     1. Add to appropriate version directory
     2. Update version's README
     3. Add tests
     4. Document changes
     5. Update configuration if needed

3. **Code Review Checklist**
   - [ ] Follows PEP 8 guidelines
   - [ ] Includes type hints
   - [ ] Has comprehensive tests
   - [ ] Documentation is complete
   - [ ] No linting errors
   - [ ] All tests pass
   - [ ] Performance impact considered

### Testing

1. **Test Structure**
   ```
   tests/
   ├── common/
   │   ├── test_llm_base.py
   │   ├── test_schema_utils.py
   │   └── test_io_utils.py
   ├── v1/
   │   ├── test_llm_client.py
   │   └── test_main.py
   └── v2/
       ├── test_llm_client.py
       └── test_main.py
   ```

2. **Running Tests**
   ```bash
   # Run all tests
   pytest

   # Run specific test file
   pytest tests/v1/test_llm_client.py

   # Run with coverage
   pytest --cov=llm_pipeline

   # Run specific test
   pytest tests/v1/test_llm_client.py::TestLLMClient::test_phone_extraction
   ```

3. **Test Guidelines**
   - Write unit tests for all new features
   - Include edge cases and error conditions
   - Mock external API calls
   - Use fixtures for common test data
   ```python
   @pytest.fixture
   def sample_text():
       return "Contact us at +1 (800) 555-1234"

   def test_phone_extraction(sample_text):
       client = V1LLMClient()
       result = client.generate(sample_text)
       assert "+1 (800) 555-1234" in result["Sales"]
   ```

### Versioning

1. **Version Number Format**
   - Follow semantic versioning (MAJOR.MINOR.PATCH)
   - MAJOR: Breaking changes
   - MINOR: New features, backward compatible
   - PATCH: Bug fixes, backward compatible

2. **Release Process**
   ```bash
   # Update version in setup.py
   version = "1.2.3"

   # Create release branch
   git checkout -b release/v1.2.3

   # Update CHANGELOG.md
   # Run final tests
   pytest

   # Create release tag
   git tag -a v1.2.3 -m "Release v1.2.3"
   git push origin v1.2.3
   ```

3. **Changelog Format**
   ```markdown
   # Changelog

   ## [1.2.3] - 2024-03-20
   ### Added
   - New feature X
   - Support for Y

   ### Changed
   - Improved performance of Z
   - Updated documentation

   ### Fixed
   - Bug in feature A
   - Issue with B
   ```

## Troubleshooting

### Common Issues and Solutions

1. **LLM API Issues**
   - **Problem**: API requests failing with authentication errors
   - **Solution**: 
     ```bash
     # Verify API key is set
     echo $LLM_API_KEY
     
     # Check API endpoint in config.py
     cat config.py | grep LLM_API_URL
     ```
   - **Problem**: Rate limiting errors
   - **Solution**: 
     ```python
     # Adjust rate limits in config.py
     API_RATE_LIMIT = 100  # Reduce requests per minute
     MAX_RETRIES = 5       # Increase retry attempts
     ```

2. **File and Directory Issues**
   - **Problem**: Missing input/output directories
   - **Solution**: 
     ```python
     # Add to your script
     from pathlib import Path
     
     # Create required directories
     for dir_path in [PAGES_DIR, OCR_DIR, OUTPUTS_V1, OUTPUTS_V2]:
         Path(dir_path).mkdir(parents=True, exist_ok=True)
     ```
   - **Problem**: Permission denied errors
   - **Solution**: 
     ```bash
     # Check directory permissions
     ls -la /path/to/directory
     
     # Fix permissions if needed
     chmod 755 /path/to/directory
     ```

3. **OCR Processing Issues**
   - **Problem**: Tesseract not found
   - **Solution**: 
     ```bash
     # Verify Tesseract installation
     tesseract --version
     
     # Reinstall if needed
     # Windows:
     # Download from https://github.com/UB-Mannheim/tesseract/wiki
     
     # Linux:
     sudo apt-get install tesseract-ocr
     
     # MacOS:
     brew install tesseract
     ```
   - **Problem**: Poor OCR quality
   - **Solution**: 
     ```python
     # Adjust OCR settings in config.py
     OCR_DPI = 300        # Increase DPI for better quality
     OCR_LANGUAGES = ["eng"]  # Specify required languages
     ```

4. **Memory and Performance Issues**
   - **Problem**: Out of memory errors
   - **Solution**: 
     ```python
     # Reduce batch size and chunk size
     BATCH_SIZE = 10      # Smaller batches
     CHUNK_SIZE = 500     # Smaller chunks
     ```
   - **Problem**: Slow processing
   - **Solution**: 
     ```python
     # Enable caching
     CACHE_ENABLED = True
     CACHE_DIR = "cache"
     
     # Use GPU acceleration if available
     USE_GPU = True
     ```

5. **Output Format Issues**
   - **Problem**: Invalid JSON output
   - **Solution**: 
     ```python
     # Validate output schema
     from llm_pipeline.common.schema_utils import validate_output
     
     result = validate_output(output_data)
     ```
   - **Problem**: Missing metadata
   - **Solution**: 
     ```python
     # Enable metadata tracking
     INCLUDE_METADATA = True
     SAVE_INTERMEDIATE = True
     ```

### Debugging Tips

1. **Enable Debug Logging**
   ```python
   # In config.py
   LOG_LEVEL = "DEBUG"
   LOG_FILE = "pipeline.log"
   
   # In your code
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test Individual Components**
   ```bash
   # Test LLM client
   python -m llm_pipeline.v1.test_prompt
   
   # Test OCR processing
   python -m llm_pipeline.common.ocr_utils test_image.jpg
   
   # Test schema validation
   python -m llm_pipeline.common.schema_utils test_output.json
   ```

3. **Monitor Resource Usage**
   ```bash
   # Check memory usage
   top -p $(pgrep -f "python.*llm_pipeline")
   
   # Monitor disk space
   df -h /path/to/output/directory
   ```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/your-repo/issues) page
2. Review the [API Documentation](https://api-docs.example.com)
3. Join our [Discord Community](https://discord.gg/your-server)
4. Contact support at support@example.com

## Contributing
