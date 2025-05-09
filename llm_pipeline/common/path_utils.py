"""Common path utilities for the LLM pipeline."""

import os
from pathlib import Path

def get_project_root() -> Path:
    """Get the absolute path to the project root directory.
    
    Returns:
        Path object pointing to the project root.
    """
    # Start from the current file's directory
    current_dir = Path(__file__).parent
    
    # Navigate up to the project root (where llm_pipeline directory is)
    while current_dir.name != "llm_pipeline" and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    
    # Go up one more level to get to the project root
    return current_dir.parent 