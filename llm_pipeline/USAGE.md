# LLM Pipeline Usage Guide

This document provides detailed information about the command-line arguments and usage examples for the LLM pipeline.

## Overview

The LLM pipeline comes in two versions (v1 and v2) and can process text content, OCR results, or both. It uses different LLM APIs (llama or mixtral) to extract structured information from the content.

## Command-Line Arguments

### Common Arguments (Both v1 and v2)

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--debug` | Enable debug logging | False | - |
| `--api-type` | Type of LLM API to use | 'llama' | 'llama', 'mixtral' |
| `--data-dir` | Custom data directory path | - | - |
| `--run-name` | Optional name for this run | - | - |

### v2-Specific Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--limit` | Limit the number of files to process | - | - |
| `--overwrite` | Reprocess already completed hostnames | False | - |
| `--source` | Source type to process | 'text' | 'text', 'ocr', 'ocr+text' |
| `--hostnames` | Comma-separated list of hostnames to process | - | - |
| `--preview` | Print the first processed result | False | - |

## Usage Examples

### Basic Usage

1. Process files using v1 pipeline:
```bash
python -m llm_pipeline.v1.main --api-type mixtral --run-name "example_run"
```

2. Process files using v2 pipeline:
```bash
python -m llm_pipeline.v2.main --source ocr+text
```

### Source Types (v2 Only)

1. Process text content only:
```bash
python -m llm_pipeline.v2.main --source text
```

2. Process OCR results only:
```bash
python -m llm_pipeline.v2.main --source ocr
```

3. Process both text and OCR results:
```bash
python -m llm_pipeline.v2.main --source ocr+text
```

### API Selection

1. Use Llama API (default):
```bash
python -m llm_pipeline.v2.main --api-type llama
```

2. Use Mixtral API:
```bash
python -m llm_pipeline.v2.main --api-type mixtral
```

### Processing Control (v2 Only)

1. Limit the number of files to process:
```bash
python -m llm_pipeline.v2.main --limit 50
```

2. Process specific hostnames:
```bash
python -m llm_pipeline.v2.main --hostnames "example.com,test.com"
```

3. Overwrite existing results:
```bash
python -m llm_pipeline.v2.main --overwrite
```

### Debugging and Testing

1. Enable debug logging:
```bash
python -m llm_pipeline.v2.main --debug
```

2. Preview the first result (v2 only):
```bash
python -m llm_pipeline.v2.main --preview
```

### Test Prompts

1. Test v1 prompt:
```bash
python -m llm_pipeline.v1.test_prompt
```

2. Test v2 prompt with input file:
```bash
python -m llm_pipeline.v2.test_prompt --input-file path/to/text.txt
```

### Combined Examples

1. Full featured v2 example:
```bash
python -m llm_pipeline.v2.main \
    --source ocr+text \
    --api-type mixtral \
    --limit 50 \
    --debug \
    --run-name "test_run"
```

2. Process specific hostnames with custom data directory:
```bash
python -m llm_pipeline.v2.main \
    --source text \
    --hostnames "example.com,test.com" \
    --data-dir "./custom_data" \
    --api-type llama
```

## Output Structure

The LLM pipeline organizes its output in the following structure:

```
outputs/
├── v1/                # v1 pipeline outputs
│   └── {run_name}/   # Run-specific outputs
│       ├── logs/     # Log files
│       └── outputs/  # Processed results
└── v2/                # v2 pipeline outputs
    └── {run_name}/   # Run-specific outputs
        ├── logs/     # Log files
        └── outputs/  # Processed results
```

## Environment Variables

The LLM pipeline can be configured using environment variables:

- `LLM_API_URL`: URL for the LLM API endpoint
- `DEFAULT_MODEL_NAME`: Default model to use (e.g., "gpt-4", "gpt-3.5-turbo")
- `API_KEY`: API key for the LLM service
- `ENVIRONMENT`: Set to "prod" or "dev" to load appropriate configuration 