"""Utility functions for the v2 LLM pipeline."""

from pathlib import Path
from typing import List, Set
from llm_pipeline.config import PAGES_DIR, PROMPT_V2_PATH
from llm_pipeline.common.io_utils import load_text

def get_text_files(pattern: str = "**/text.txt") -> List[Path]:
    """Get all files matching the pattern from PAGES_DIR.
    
    Args:
        pattern: Glob pattern to match files. Defaults to "**/text.txt".
    
    Returns:
        List of Path objects matching the pattern.
    """
    return list(PAGES_DIR.glob(pattern))

def get_hostname_from_path(text_file: Path) -> str:
    """Extract the hostname from the text file path.
    
    Args:
        text_file: Path to the text file.
        
    Returns:
        The hostname (parent directory name).
        
    Raises:
        ValueError: If the path is malformed or parent directory is missing.
    """
    if not text_file.parent.exists():
        raise ValueError(f"Parent directory does not exist for path: {text_file}")
    
    parent_name = text_file.parent.name
    if not parent_name:
        raise ValueError(f"Parent directory has no name for path: {text_file}")
        
    return parent_name

def load_prompt() -> str:
    """Load the v2 phone extraction prompt from file."""
    return load_text(PROMPT_V2_PATH)

def get_output_path(hostname: str, output_dir: Path) -> Path:
    """Generate the output JSON path for a given hostname.
    
    Args:
        hostname: The hostname to generate the output path for.
        output_dir: Directory where the output file should be saved.
        
    Returns:
        Path object pointing to the output JSON file.
    """
    return output_dir / f"{hostname}.json"

def get_all_hostnames(files: List[Path]) -> Set[str]:
    """Extract unique hostnames from a list of file paths.
    
    Args:
        files: List of file paths to extract hostnames from.
        
    Returns:
        Set of unique hostnames.
        
    Raises:
        ValueError: If any path is malformed (see get_hostname_from_path).
    """
    return {get_hostname_from_path(file) for file in files} 