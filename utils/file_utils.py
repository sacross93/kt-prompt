"""
File utility functions
"""
import os
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.exceptions import FileProcessingError

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure directory exists, create if not"""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        raise FileProcessingError(f"Failed to create directory {directory_path}: {e}")

def read_text_file(file_path: str) -> str:
    """Read text file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileProcessingError(f"File not found: {file_path}")
    except Exception as e:
        raise FileProcessingError(f"Failed to read file {file_path}: {e}")

def write_text_file(file_path: str, content: str) -> None:
    """Write content to text file"""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise FileProcessingError(f"Failed to write file {file_path}: {e}")

def append_text_file(file_path: str, content: str) -> None:
    """Append content to text file"""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise FileProcessingError(f"Failed to append to file {file_path}: {e}")

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read CSV file and return list of dictionaries"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        raise FileProcessingError(f"CSV file not found: {file_path}")
    except Exception as e:
        raise FileProcessingError(f"Failed to read CSV file {file_path}: {e}")

def write_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """Write data to JSON file"""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise FileProcessingError(f"Failed to write JSON file {file_path}: {e}")

def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileProcessingError(f"JSON file not found: {file_path}")
    except Exception as e:
        raise FileProcessingError(f"Failed to read JSON file {file_path}: {e}")

def get_next_version_filename(base_path: str, extension: str = ".txt") -> str:
    """Get next version filename (e.g., prompt_v1.txt -> prompt_v2.txt)"""
    directory = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    
    version = 1
    while True:
        filename = f"{base_name}_v{version}{extension}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return full_path
        version += 1

def backup_file(file_path: str, backup_suffix: str = None) -> str:
    """Create backup of file"""
    if not os.path.exists(file_path):
        raise FileProcessingError(f"File to backup does not exist: {file_path}")
    
    if backup_suffix is None:
        backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    directory = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    extension = os.path.splitext(file_path)[1]
    
    backup_path = os.path.join(directory, f"{base_name}_backup_{backup_suffix}{extension}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        return backup_path
    except Exception as e:
        raise FileProcessingError(f"Failed to backup file {file_path}: {e}")

def list_files_with_pattern(directory: str, pattern: str) -> List[str]:
    """List files in directory matching pattern"""
    import glob
    try:
        pattern_path = os.path.join(directory, pattern)
        return glob.glob(pattern_path)
    except Exception as e:
        raise FileProcessingError(f"Failed to list files in {directory}: {e}")