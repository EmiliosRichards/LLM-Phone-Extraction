# Project Structure

This project implements a scalable, modular pipeline for extracting and analyzing phone numbers from web pages using LLMs. The structure supports multiple versions of the pipeline, allowing for iterative improvements while maintaining backward compatibility.

## Directory Structure

```
.
├── README.md                # Project overview and setup instructions
├── requirements.txt         # Python dependencies
├── structure.md            # Project structure documentation
├── tree_output.txt         # Directory tree output
│
├── data/                   # Processed data storage
│   ├── images/            # Screenshots and images
│   ├── llm_outputs/       # v1 pipeline outputs
│   ├── llm_outputs_v2/    # v2 pipeline outputs
│   └── pages/             # Raw webpage text
│
├── docs/                   # Documentation
│   └── structure.md       # Project structure documentation
│
├── llm_pipeline/          # LLM processing pipelines
│   ├── config.py          # Central configuration
│   ├── common/            # Shared utilities
│   │   ├── io_utils.py    # File I/O operations
│   │   ├── llm_base.py    # Base LLM functionality
│   │   ├── log.py         # Logging setup
│   │   ├── schema_utils.py # Schema validation utilities
│   │   ├── text_utils.py  # Text processing utilities
│   │   └── __init__.py
│   │
│   ├── v1/               # Basic phone number extraction
│   │   ├── llm_client.py # LLM API interaction
│   │   ├── main.py       # Entry point
│   │   ├── test_prompt.py # Prompt testing
│   │   ├── utils.py      # Utilities
│   │   └── __init__.py
│   │
│   └── v2/               # Enhanced extraction pipeline
│       ├── llm_client.py # Structured LLM interaction
│       ├── main.py       # Entry point
│       ├── schema.md     # Output schema definition
│       ├── test_prompt.py # Prompt testing
│       ├── utils.py      # Utilities
│       └── __init__.py
│
├── logs/                  # Application logs
│
├── prompts/              # LLM prompt templates
│   ├── phone_extraction_prompt.txt    # v1 prompt
│   └── phone_extraction_prompt_v2.txt # v2 prompt
│
└── scraper/             # Web scraping and data collection
    ├── config.py        # Scraper configuration
    ├── main.py          # Entry point
    ├── ocr.py           # OCR processing
    ├── rate_limiter.py  # Rate limiting utilities
    ├── scraper.py       # Core scraping functionality
    ├── utils.py         # Utilities
    ├── __init__.py
    └── tests/           # Scraper test suite
        ├── test_ocr.py
        ├── test_rate_limiter.py
        ├── test_rate_limiter_concurrency.py
        └── test_utils.py
```

## Components

### Scraper (`scraper/`)
- Extracts text content from web pages
- Captures screenshots for visual reference
- Performs OCR on images to extract text
- Includes rate limiting for API calls
- Comprehensive test suite for core functionality
- Components:
  - `main.py`: Entry point
  - `scraper.py`: Core scraping functionality
  - `ocr.py`: OCR processing
  - `rate_limiter.py`: API rate limiting
  - `utils.py`: Utility functions
  - `config.py`: Scraper configuration
  - `tests/`: Test suite for all components

### LLM Pipeline v1 (`llm_pipeline/v1/`)
- Basic phone number extraction pipeline
- Simple JSON output format
- Components:
  - `main.py`: Entry point with CLI arguments
  - `llm_client.py`: LLM API interaction
  - `utils.py`: File handling utilities
  - `test_prompt.py`: Prompt testing tool
- Output: Basic JSON with extracted phone numbers
- Uses `data/llm_outputs/` for results

### LLM Pipeline v2 (`llm_pipeline/v2/`)
- Enhanced structured extraction pipeline
- Rich metadata and confidence scores
- Components:
  - `main.py`: Entry point with CLI arguments
  - `llm_client.py`: Structured LLM interaction
  - `utils.py`: File handling utilities
  - `test_prompt.py`: Prompt testing tool
  - `schema.md`: Output schema definition
- Output: Structured JSON with:
  - Phone numbers with categories
  - Confidence scores
  - Context snippets
  - Processing metadata
- Uses `data/llm_outputs_v2/` for results

### Common Utilities (`llm_pipeline/common/`)
- Shared functionality across pipelines
- Components:
  - `io_utils.py`: File I/O operations
  - `llm_base.py`: Base LLM functionality
  - `log.py`: Logging setup
  - `schema_utils.py`: Schema validation
  - `text_utils.py`: Text processing
- Reduces code duplication
- Ensures consistent behavior

### Data Storage (`data/`)
- `pages/`: Raw text extracted from web pages
  - Organized by hostname
  - Each page has a `text.txt` file
- `images/`: Screenshots and images
  - Used for visual reference
  - Supports OCR processing
- `llm_outputs/`: v1 pipeline results
- `llm_outputs_v2/`: v2 pipeline results

### Prompts (`prompts/`)
- `phone_extraction_prompt.txt`: v1 prompt template
- `phone_extraction_prompt_v2.txt`: v2 structured prompt
- Templates use `{webpage_text}` placeholder
- Designed for consistent LLM responses

### Configuration
- `llm_pipeline/config.py`: Central LLM pipeline configuration
- `scraper/config.py`: Scraper-specific configuration
- Defines all directory paths
- LLM API settings
- Model parameters
- Ensures consistent paths across pipelines

### Shared Components

#### Configuration (`config.py`)
The central configuration file manages all pipeline settings and ensures consistency across versions.

```python
# Key configuration sections
PATHS = {
    'data': 'data/',
    'outputs': {
        'v1': 'data/llm_outputs/',
        'v2': 'data/llm_outputs_v2/'
    }
}

LLM_SETTINGS = {
    'v1': {
        'model': 'gpt-3.5-turbo',
        'temperature': 0.1
    },
    'v2': {
        'model': 'gpt-4',
        'temperature': 0.2
    }
}
```

**Features**:
- Centralized path management
- Version-specific LLM settings
- Environment variable handling
- Logging configuration
- Rate limiting parameters

**Usage in Pipelines**:
- v1: Uses basic settings for quick processing
- v2: Leverages enhanced settings for structured output
- Both versions share common paths and base configurations

#### Common Utilities (`common/`)

1. **io_utils.py**
   - File I/O operations for all pipeline versions
   - Handles reading/writing of:
     - Raw webpage text
     - LLM outputs
     - Configuration files
   - Implements version-specific output formatting
   ```python
   def read_webpage(hostname: str) -> str:
       """Read webpage text with version-agnostic path handling."""
   
   def write_output(data: dict, version: str):
       """Write output with version-specific formatting."""
   ```

2. **llm_base.py**
   - Base LLM client functionality
   - Common API interaction patterns
   - Error handling and retries
   - Rate limiting implementation
   ```python
   class BaseLLMClient:
       """Base class for LLM interactions."""
       def process_prompt(self, prompt: str) -> dict:
           """Process prompt with version-specific handling."""
   ```

3. **log.py**
   - Centralized logging setup
   - Version-specific log formatting
   - Log rotation and management
   - Error tracking across versions
   ```python
   def setup_logging(version: str):
       """Configure logging for specific pipeline version."""
   
   def log_error(error: Exception, version: str):
       """Log errors with version context."""
   ```

4. **schema_utils.py**
   - JSON Schema validation
   - Version-specific schema management
   - Output format verification
   - Schema migration utilities
   ```python
   def validate_output(data: dict, version: str) -> bool:
       """Validate output against version-specific schema."""
   
   def migrate_schema(data: dict, from_version: str, to_version: str):
       """Migrate data between schema versions."""
   ```

5. **text_utils.py**
   - Text processing utilities
   - Phone number formatting
   - Context extraction
   - Text cleaning and normalization
   ```python
   def extract_context(text: str, position: int) -> str:
       """Extract context around phone numbers."""
   
   def normalize_phone(phone: str) -> str:
       """Normalize phone number format."""
   ```

#### Maintaining Consistency

1. **Version Compatibility**
   - Common utilities support both v1 and v2
   - Backward compatibility maintained
   - Version-specific features isolated
   - Shared functionality maximized

2. **Error Handling**
   - Consistent error types across versions
   - Centralized error logging
   - Version-specific error recovery
   - Common retry mechanisms

3. **Configuration Management**
   - Single source of truth in `config.py`
   - Version-specific settings isolated
   - Common settings shared
   - Environment-based configuration

4. **Code Reuse**
   - Common utilities reduce duplication
   - Consistent interfaces across versions
   - Shared validation logic
   - Common helper functions

5. **Testing**
   - Shared test utilities
   - Version-specific test cases
   - Common test fixtures
   - Integration test support

## Pipeline Features

### v1 vs v2 Comparison

#### Key Improvements in v2
- **Structured Output**: v2 provides a well-defined schema with categorized phone numbers and rich metadata
- **Confidence Scoring**: Each extracted phone number includes a confidence score to help assess reliability
- **Context Preservation**: Captures surrounding text context for each phone number
- **Metadata Tracking**: Records processing timestamps, source information, and pipeline version
- **Schema Validation**: Built-in validation ensures consistent output format
- **Enhanced Error Handling**: More robust error recovery and logging

#### When to Use Each Version
- **Use v1 when**:
  - You need simple, quick phone number extraction
  - Processing speed is prioritized over metadata
  - Working with straightforward web pages
  - Basic JSON output is sufficient

- **Use v2 when**:
  - You need detailed metadata and confidence scores
  - Context around phone numbers is important
  - Working with complex or ambiguous content
  - Requiring structured, validated output
  - Need to track processing history

Both versions can be used in parallel, with v1 providing quick initial results while v2 offers more detailed analysis.

### Command Line Interface
Both v1 and v2 pipelines support:
- `--limit`: Process N files
- `--overwrite`: Reprocess existing files
- `--source`: Choose input source:
  - `text`: Raw text only
  - `ocr`: OCR text only
  - `ocr+text`: Both sources

### Scalability
- Modular design allows easy addition of new pipeline versions
- Shared utilities reduce maintenance overhead
- Consistent configuration management
- Support for multiple data sources

#### Best Practices for Adding New Versions
1. **Directory Structure**
   ```
   llm_pipeline/
   ├── v3/                    # New version directory
   │   ├── main.py           # Entry point
   │   ├── llm_client.py     # Version-specific LLM logic
   │   ├── utils.py          # Version-specific utilities
   │   ├── test_prompt.py    # Prompt testing
   │   ├── schema.md         # Output schema definition
   │   └── __init__.py
   ```

2. **Version Naming**
   - Use semantic versioning (v1, v2, v3)
   - Maintain backward compatibility within major versions
   - Document breaking changes in README.md

3. **Configuration Management**
   - Add version-specific settings to `config.py`
   - Use environment variables for sensitive data
   - Maintain consistent naming conventions
   ```python
   # config.py
   V3_SETTINGS = {
       'model': 'gpt-4',
       'temperature': 0.2,
       'max_tokens': 1000
   }
   ```

4. **Output Organization**
   - Create dedicated output directory
   - Use consistent naming patterns
   - Implement version-specific schemas
   ```
   data/
   ├── llm_outputs/     # v1 outputs
   ├── llm_outputs_v2/  # v2 outputs
   └── llm_outputs_v3/  # v3 outputs
   ```

### Extensibility
- Easy to add new features:
  - New prompt templates
  - Additional data sources
  - Enhanced output formats
  - Custom processing steps

#### Adding New Features

1. **Prompt Templates**
   - Place new templates in `prompts/`
   - Follow naming convention: `feature_name_prompt.txt`
   - Include version suffix if version-specific
   ```
   prompts/
   ├── phone_extraction_prompt.txt
   ├── phone_extraction_prompt_v2.txt
   └── new_feature_prompt_v3.txt
   ```

2. **Common Utilities**
   - Add shared functionality to `common/`
   - Create new utility modules as needed
   - Maintain consistent interface
   ```python
   # common/new_feature_utils.py
   def process_feature(data):
       """Process new feature data.
       
       Args:
           data: Input data to process
           
       Returns:
           Processed data
       """
       # Implementation
   ```

3. **Schema Updates**
   - Define schemas in `schema.md`
   - Use JSON Schema format
   - Include validation rules
   ```json
   {
     "type": "object",
     "properties": {
       "new_feature": {
         "type": "string",
         "description": "New feature data"
       }
     }
   }
   ```

4. **Testing**
   - Add unit tests for new features
   - Include integration tests
   - Test across versions
   ```
   tests/
   ├── test_new_feature.py
   └── test_integration.py
   ```

#### Maintaining Consistency

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use consistent docstring format
   - Maintain similar function signatures

2. **Error Handling**
   - Use common error types
   - Implement consistent logging
   - Follow error recovery patterns

3. **Documentation**
   - Update README.md for new features
   - Document API changes
   - Include usage examples

4. **Version Control**
   - Create feature branches
   - Use meaningful commit messages
   - Review changes across versions

## Usage

### Basic Processing

1. Run the scraper to collect data:
```bash
python -m scraper.main
```

2. Process with v1 pipeline:
```bash
python -m llm_pipeline.v1.main --limit 50 --source text
```

3. Process with v2 pipeline:
```bash
python -m llm_pipeline.v2.main --limit 50 --source ocr+text --overwrite
```

### Advanced Usage

#### Batch Processing

1. **Process Large Datasets**
```bash
# Process 1000 pages with v2, using both text and OCR
python -m llm_pipeline.v2.main \
    --limit 1000 \
    --source ocr+text \
    --batch-size 50 \
    --output-dir data/custom_outputs \
    --log-level INFO
```

2. **Resume Failed Jobs**
```bash
# Resume processing from last successful point
python -m llm_pipeline.v2.main \
    --resume \
    --checkpoint data/checkpoints/last_run.json \
    --retry-failed
```

3. **Parallel Processing**
```bash
# Process multiple batches in parallel
python -m llm_pipeline.v2.main \
    --parallel \
    --workers 4 \
    --batch-size 25
```

#### Error Handling and Retries

1. **Configure Retry Behavior**
```python
# config.py
RETRY_SETTINGS = {
    'max_retries': 3,
    'retry_delay': 5,  # seconds
    'backoff_factor': 2,
    'retry_on': [
        'rate_limit',
        'timeout',
        'connection_error'
    ]
}
```

2. **Handle Specific Errors**
```bash
# Process with custom error handling
python -m llm_pipeline.v2.main \
    --error-handling strict \
    --retry-strategy exponential \
    --max-retries 5
```

3. **Log Error Details**
```bash
# Enable detailed error logging
python -m llm_pipeline.v2.main \
    --log-level DEBUG \
    --error-log data/error_logs/run_2024_03_21.log
```

#### Monitoring and Management

1. **Progress Tracking**
```bash
# Monitor progress in real-time
python -m llm_pipeline.v2.main \
    --progress-bar \
    --status-interval 10 \
    --output-format json
```

2. **Resource Monitoring**
```bash
# Monitor system resources
python -m llm_pipeline.v2.main \
    --monitor-resources \
    --memory-limit 4G \
    --cpu-limit 80
```

3. **Job Management**
```bash
# Save and load job state
python -m llm_pipeline.v2.main \
    --save-state data/job_states/current.json \
    --load-state data/job_states/previous.json
```

#### Output Management

1. **Organize Outputs**
```bash
# Structure outputs by date and source
python -m llm_pipeline.v2.main \
    --output-structure date_source \
    --date-format %Y%m%d \
    --compress-outputs
```

2. **Validate Results**
```bash
# Validate outputs against schema
python -m llm_pipeline.v2.main \
    --validate-outputs \
    --schema-path llm_pipeline/v2/schema.md \
    --strict-validation
```

3. **Export Results**
```bash
# Export to different formats
python -m llm_pipeline.v2.main \
    --export-format csv \
    --export-path data/exports/results.csv \
    --include-metadata
```

### Monitoring Long-Running Jobs

1. **Check Job Status**
```bash
# View current job status
python -m llm_pipeline.v2.main --status

# Output example:
# Progress: 45% (450/1000)
# Success: 445
# Failed: 5
# Pending: 550
# Estimated time remaining: 2h 15m
```

2. **Monitor System Resources**
```bash
# View resource usage
python -m llm_pipeline.v2.main --monitor

# Output example:
# CPU Usage: 75%
# Memory Usage: 2.3GB/4GB
# Disk Usage: 1.2GB
# Network: 5MB/s
```

3. **Handle Interruptions**
```bash
# Gracefully stop processing
python -m llm_pipeline.v2.main --stop

# Resume from last checkpoint
python -m llm_pipeline.v2.main --resume
```

### Best Practices

1. **Data Management**
   - Use `--batch-size` to control memory usage
   - Enable `--compress-outputs` for large datasets
   - Implement regular checkpoints with `--save-state`

2. **Error Handling**
   - Set appropriate `--max-retries`
   - Use `--error-log` for debugging
   - Enable `--validate-outputs` for quality control

3. **Resource Optimization**
   - Monitor system resources with `--monitor-resources`
   - Adjust `--workers` based on available CPU
   - Use `--memory-limit` to prevent OOM errors

4. **Output Organization**
   - Structure outputs by date/source
   - Enable compression for large datasets
   - Validate outputs against schema

## Troubleshooting

### Common Issues and Solutions

#### API and Rate Limiting

1. **API Rate Limit Errors**
   ```
   Error: Rate limit exceeded. Please try again in 60 seconds.
   ```
   **Solutions**:
   - Use `--rate-limit` to set appropriate delays
   - Enable `--retry-on-rate-limit`
   - Implement exponential backoff:
     ```bash
     python -m llm_pipeline.v2.main \
         --retry-strategy exponential \
         --backoff-factor 2
     ```

2. **API Authentication Failures**
   ```
   Error: Invalid API key or authentication failed
   ```
   **Solutions**:
   - Verify API key in environment variables
   - Check API key permissions
   - Ensure proper key format
   - Use `--debug` flag for detailed error info

#### File System Issues

1. **Missing Input Files**
   ```
   Error: Input file not found: data/pages/example.com/text.txt
   ```
   **Solutions**:
   - Verify file paths in `config.py`
   - Check file permissions
   - Ensure scraper completed successfully
   - Use `--validate-inputs` to check before processing

2. **Disk Space Issues**
   ```
   Error: Insufficient disk space for output
   ```
   **Solutions**:
   - Enable `--compress-outputs`
   - Use `--cleanup-temp` to remove temporary files
   - Set `--max-disk-usage` to prevent overflow
   - Implement regular cleanup:
     ```bash
     python -m llm_pipeline.v2.main --cleanup-old-outputs --days 7
     ```

#### Processing Errors

1. **Memory Errors**
   ```
   Error: Out of memory while processing batch
   ```
   **Solutions**:
   - Reduce `--batch-size`
   - Enable `--stream-processing`
   - Set `--memory-limit`
   - Use `--optimize-memory` flag

2. **Timeout Errors**
   ```
   Error: Processing timeout after 300 seconds
   ```
   **Solutions**:
   - Increase `--timeout` value
   - Reduce batch size
   - Enable `--partial-results`
   - Use `--timeout-strategy continue`

#### Output Issues

1. **Invalid Output Format**
   ```
   Error: Output validation failed against schema
   ```
   **Solutions**:
   - Check schema version compatibility
   - Use `--validate-outputs` to identify issues
   - Enable `--strict-validation` for detailed errors
   - Review schema in `llm_pipeline/v2/schema.md`

2. **Missing Metadata**
   ```
   Warning: Required metadata fields missing
   ```
   **Solutions**:
   - Enable `--include-metadata`
   - Check metadata configuration
   - Use `--metadata-template` for custom fields
   - Verify schema requirements

### Debugging Tools

1. **Enable Debug Mode**
   ```bash
   python -m llm_pipeline.v2.main --debug
   ```
   - Shows detailed error messages
   - Logs API requests/responses
   - Tracks memory usage
   - Records processing steps

2. **Check System Status**
   ```bash
   python -m llm_pipeline.v2.main --system-check
   ```
   - Verifies file permissions
   - Checks API connectivity
   - Validates configuration
   - Tests disk space

3. **View Logs**
   ```bash
   # View recent errors
   python -m llm_pipeline.v2.main --show-errors
   
   # View processing history
   python -m llm_pipeline.v2.main --show-history
   ```

### Recovery Procedures

1. **Resume Failed Job**
   ```bash
   # Resume from last checkpoint
   python -m llm_pipeline.v2.main \
       --resume \
       --checkpoint data/checkpoints/last_run.json \
       --retry-failed
   ```

2. **Recover Corrupted Outputs**
   ```bash
   # Validate and repair outputs
   python -m llm_pipeline.v2.main \
       --repair-outputs \
       --backup-dir data/backups
   ```

3. **Clean Up and Restart**
   ```bash
   # Clean temporary files and restart
   python -m llm_pipeline.v2.main \
       --cleanup \
       --restart
   ```

### Getting Help

1. **Check Documentation**
   - Review `README.md` for setup instructions
   - Consult `docs/` for detailed guides
   - Check schema documentation
   - Review error code reference

2. **Enable Verbose Logging**
   ```bash
   python -m llm_pipeline.v2.main \
       --log-level DEBUG \
       --log-file data/logs/debug.log
   ```

3. **Report Issues**
   - Include error logs
   - Provide configuration details
   - Share reproduction steps
   - Note system environment

## Development

### Adding New Features

#### Code Standards

1. **Python Style Guide**
   - Follow PEP 8 guidelines
   - Use type hints for all functions
   - Document all public APIs
   ```python
   def process_data(data: Dict[str, Any], version: str) -> Dict[str, Any]:
       """Process input data according to pipeline version.
       
       Args:
           data: Input data dictionary
           version: Pipeline version ('v1' or 'v2')
           
       Returns:
           Processed data dictionary
           
       Raises:
           ValueError: If version is invalid
           ProcessingError: If data processing fails
       """
   ```

2. **Module Organization**
   - One class/function per file
   - Clear module hierarchy
   - Consistent naming conventions
   ```
   llm_pipeline/
   ├── v3/
   │   ├── feature_name/
   │   │   ├── __init__.py
   │   │   ├── processor.py
   │   │   └── utils.py
   │   └── main.py
   │
   └── v2/
       └── main.py
   ```

3. **Error Handling**
   - Use custom exception classes
   - Implement proper error recovery
   - Log detailed error information
   ```python
   class PipelineError(Exception):
       """Base exception for pipeline errors."""
       pass

   class ProcessingError(PipelineError):
       """Raised when data processing fails."""
       pass
   ```

#### Documentation Updates

1. **Code Documentation**
   - Docstrings for all modules
   - Function/method documentation
   - Type hints and return values
   - Example usage

2. **Project Documentation**
   - Update `README.md`
   - Add feature documentation
   - Update schema documentation
   - Document breaking changes

3. **API Documentation**
   - Document public interfaces
   - Provide usage examples
   - List all parameters
   - Document return values

### Testing Guidelines

#### Unit Testing

1. **Test Structure**
   ```
   tests/
   ├── unit/
   │   ├── test_feature_name.py
   │   └── test_utils.py
   ├── integration/
   │   └── test_pipeline.py
   └── conftest.py
   ```

2. **Test Coverage**
   ```python
   # test_feature_name.py
   def test_feature_processing():
       """Test feature processing functionality."""
       input_data = {...}
       expected_output = {...}
       result = process_feature(input_data)
       assert result == expected_output
   ```

3. **Test Cases**
   - Happy path scenarios
   - Edge cases
   - Error conditions
   - Performance tests

#### Integration Testing

1. **Pipeline Testing**
   ```python
   def test_pipeline_integration():
       """Test full pipeline integration."""
       # Setup test data
       # Run pipeline
       # Verify results
       # Clean up
   ```

2. **Version Compatibility**
   ```python
   def test_version_compatibility():
       """Test compatibility between versions."""
       # Test v1 to v2 migration
       # Test v2 to v3 migration
       # Verify data consistency
   ```

### Version Management

#### Creating New Versions

1. **Version Structure**
   ```
   llm_pipeline/
   ├── v3/
   │   ├── __init__.py
   │   ├── main.py
   │   ├── llm_client.py
   │   ├── utils.py
   │   ├── schema.md
   │   └── README.md
   ```

2. **Version Configuration**
   ```python
   # config.py
   VERSION_SETTINGS = {
       'v3': {
           'model': 'gpt-4',
           'temperature': 0.2,
           'max_tokens': 1000,
           'features': ['new_feature_1', 'new_feature_2']
       }
   }
   ```

3. **Schema Updates**
   ```json
   {
     "version": "v3",
     "changes": [
       "Added new_feature_1",
       "Modified output_format",
       "Deprecated old_feature"
     ]
   }
   ```

#### Maintaining Compatibility

1. **Backward Compatibility**
   - Support old output formats
   - Maintain API compatibility
   - Provide migration tools
   ```python
   def migrate_v2_to_v3(data: Dict) -> Dict:
       """Migrate data from v2 to v3 format."""
       # Migration logic
   ```

2. **Version Detection**
   ```python
   def detect_version(data: Dict) -> str:
       """Detect pipeline version from data."""
       if 'v3_features' in data:
           return 'v3'
       elif 'v2_metadata' in data:
           return 'v2'
       return 'v1'
   ```

3. **Feature Flags**
   ```python
   FEATURE_FLAGS = {
       'v3': {
           'new_feature_1': True,
           'new_feature_2': False
       }
   }
   ```

### Best Practices

1. **Code Review Checklist**
   - [ ] Follows style guide
   - [ ] Includes tests
   - [ ] Updated documentation
   - [ ] Backward compatible
   - [ ] Performance tested
   - [ ] Security reviewed

2. **Release Process**
   - Update version numbers
   - Update documentation
   - Run full test suite
   - Create release notes
   - Tag release in version control

3. **Performance Optimization**
   - Profile code regularly
   - Optimize bottlenecks
   - Monitor memory usage
   - Track processing times

4. **Security Considerations**
   - Validate all inputs
   - Sanitize outputs
   - Secure API keys
   - Log security events

### Development Workflow

1. **Feature Development**
   ```bash
   # Create feature branch
   git checkout -b feature/new-feature
   
   # Develop and test
   python -m pytest tests/
   
   # Update documentation
   # Create pull request
   ```

2. **Version Release**
   ```bash
   # Update version
   # Run tests
   python -m pytest tests/
   
   # Build documentation
   # Create release
   git tag v3.0.0
   ```

3. **Maintenance**
   - Regular dependency updates
   - Security patches
   - Bug fixes
   - Performance improvements
