"""
Utility functions for the LLM pipeline.
"""

from pathlib import Path
from typing import List
from llm_pipeline.config import PAGES_DIR, PROMPT_V1_PATH
from llm_pipeline.common.io_utils import load_text

def get_project_root() -> Path:
    """Get the absolute path to the project root directory.
    
    Returns:
        Path object pointing to the project root.
        
    Examples:
        >>> get_project_root()
        PosixPath('/path/to/project/root')
    """
    # Assuming this file is in llm_pipeline/v1/
    return Path(__file__).parent.parent.parent

def get_text_files(pattern: str = "**/text.txt") -> List[Path]:
    """Get all files matching the given glob pattern from PAGES_DIR.
    
    Args:
        pattern: Glob pattern to match files. Defaults to "**/text.txt".
    
    Returns:
        List of Path objects matching the pattern.
        
    Examples:
        >>> get_text_files()
        [PosixPath('data/pages/example.com/text.txt'), ...]
        >>> get_text_files("**/ocr.txt")
        [PosixPath('data/pages/example.com/ocr.txt'), ...]
    """
    return list(PAGES_DIR.glob(pattern))

def get_hostname_from_path(text_file: Path) -> str:
    """Extract the hostname from the text file path.
    
    Args:
        text_file: Path to the text file.
        
    Returns:
        The hostname extracted from the parent directory name.
        
    Raises:
        ValueError: If the file path has no parent or the parent has no name.
        
    Examples:
        >>> get_hostname_from_path(Path("data/pages/example.com/text.txt"))
        'example.com'
        >>> get_hostname_from_path(Path("text.txt"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: File path has no parent directory: text.txt
    """
    if not text_file.parent:
        raise ValueError(f"File path has no parent directory: {text_file}")
    
    parent_name = text_file.parent.name
    if not parent_name:
        raise ValueError(f"Parent directory has no name: {text_file.parent}")
        
    return parent_name

def get_output_path(hostname: str, output_dir: Path) -> Path:
    """Generate the output path for a hostname's results.
    
    Args:
        hostname: The hostname to generate the output path for.
        output_dir: The directory where output files should be saved.
        
    Returns:
        Path object pointing to the output JSON file.
        
    Examples:
        >>> get_output_path("example.com", Path("output"))
        PosixPath('output/example.com.json')
    """
    return output_dir / f"{hostname}.json"

def load_prompt() -> str:
    """Load the v1 phone extraction prompt from file.
    
    Returns:
        The contents of the prompt file as a string.
        
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        IOError: If the prompt file cannot be read.
        
    Examples:
        >>> load_prompt()  # doctest: +SKIP
        'Extract phone numbers from the following text...'
        >>> # If prompt file is missing:
        >>> load_prompt()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: Prompt file not found at: /path/to/prompt.txt
    """
    try:
        return load_text(PROMPT_V1_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at: {PROMPT_V1_PATH}")
    except IOError as e:
        raise IOError(f"Failed to read prompt file at {PROMPT_V1_PATH}: {str(e)}") 