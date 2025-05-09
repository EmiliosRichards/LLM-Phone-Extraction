"""Shared configuration and constants for the LLM pipeline."""

__all__ = [
    # Directory paths
    "DATA_DIR",
    "IMAGES_DIR",
    "OCR_DIR",
    "OUTPUTS_V1",
    "OUTPUTS_V2",
    "PAGES_DIR",
    "PROJECT_ROOT",
    "PROMPTS_DIR",
    "set_custom_data_dir",
    
    # Common filenames
    "METADATA_JSON_FILENAME",
    "OCR_JSON_FILENAME",
    "OCR_TEXT_FILENAME",
    "OUTPUT_JSON_FILENAME",
    "TEXT_FILENAME",
    
    # LLM API configuration
    "DEFAULT_API_URL",
    "DEFAULT_MODEL_NAME",
    "LLM_API_URL",
    "MODEL_NAME",
    "MIXTRAL_API_URL",
    "MIXTRAL_MODEL_NAME",
    
    # Prompt paths
    "PROMPT_V1_PATH",
    "PROMPT_V2_PATH",
    
    # Functions
    "setup_directories",
]

import os
from pathlib import Path
from llm_pipeline.common.path_utils import get_project_root

# Get the project root directory
PROJECT_ROOT: Path = get_project_root()

# Directory paths - these will be updated by set_custom_data_dir if called
DATA_DIR: Path = PROJECT_ROOT / "data"
PAGES_DIR: Path = DATA_DIR / "pages"
IMAGES_DIR: Path = DATA_DIR / "images"
OCR_DIR: Path = DATA_DIR / "ocr_text"
OUTPUTS_V1: Path = DATA_DIR / "llm_outputs"
OUTPUTS_V2: Path = DATA_DIR / "llm_outputs_v2"
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"

# Common filenames
TEXT_FILENAME: str = "text.txt"
OCR_JSON_FILENAME: str = "summary.json"
OCR_TEXT_FILENAME: str = "text.txt"
OUTPUT_JSON_FILENAME: str = "output.json"
METADATA_JSON_FILENAME: str = "text.json"

# Prompt paths
PROMPT_V1_PATH: Path = PROMPTS_DIR / "phone_extraction_prompt.txt"
PROMPT_V2_PATH: Path = PROMPTS_DIR / "phone_extraction_prompt_v2.txt"

# LLM API configuration
DEFAULT_API_URL: str = "http://localhost:22760/v1/completions"
DEFAULT_MODEL_NAME: str = "Meta-Llama-3.1-70B-Instruct-GPTQ"
MIXTRAL_API_URL: str = "http://localhost:22760/api/chat"
MIXTRAL_MODEL_NAME: str = "mixtral"

# Get configuration from environment variables with fallback to defaults
LLM_API_URL: str = os.getenv("LLM_API_URL", DEFAULT_API_URL)
MODEL_NAME: str = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

def set_custom_data_dir(custom_data_dir: Path) -> None:
    """Set a custom data directory and update all related paths.
    
    Args:
        custom_data_dir: Path to the custom data directory containing scraping output
    """
    global DATA_DIR, PAGES_DIR, IMAGES_DIR, OCR_DIR, OUTPUTS_V1, OUTPUTS_V2
    
    # Convert to absolute path if relative
    if not custom_data_dir.is_absolute():
        custom_data_dir = Path.cwd() / custom_data_dir
    
    # Resolve any symlinks and normalize the path
    custom_data_dir = custom_data_dir.resolve()
    
    DATA_DIR = custom_data_dir
    # Update paths to match new structure
    PAGES_DIR = DATA_DIR / "pages"
    IMAGES_DIR = DATA_DIR / "images"
    OCR_DIR = DATA_DIR / "pages"  # OCR is now under pages/hostname/ocr
    OUTPUTS_V1 = DATA_DIR / "llm_outputs"
    OUTPUTS_V2 = DATA_DIR / "llm_outputs_v2"
    
    # Debug print the actual paths being used
    import logging
    logger = logging.getLogger("llm_pipeline.v1")
    logger.debug(f"Updated DATA_DIR to: {DATA_DIR}")
    logger.debug(f"Updated PAGES_DIR to: {PAGES_DIR}")
    logger.debug(f"Updated IMAGES_DIR to: {IMAGES_DIR}")
    logger.debug(f"Updated OCR_DIR to: {OCR_DIR}")

def setup_directories() -> None:
    """Create all required directories if they don't exist.
    
    This function will only create directories if the CREATE_PIPELINE_DIRS
    environment variable is set to "true" (case-insensitive).
    Defaults to "true" if the environment variable is not set.
    """
    if os.getenv("CREATE_PIPELINE_DIRS", "true").lower() != "true":
        return
        
    directories: list[Path] = [DATA_DIR, PAGES_DIR, IMAGES_DIR, OCR_DIR, OUTPUTS_V1, OUTPUTS_V2, PROMPTS_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True) 