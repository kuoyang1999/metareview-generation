import io
import json
import logging
import os
from typing import Dict, Optional

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def read_quantization_config(checkpoint_path: str) -> Optional[Dict]:
    """
    Read quantization config from checkpoint's README.md if it exists.
    
    Args:
        checkpoint_path: Path to the checkpoint directory containing README.md
        
    Returns:
        Dict of quantization settings if found, None otherwise
    """
    readme_path = os.path.join(checkpoint_path, "README.md")
    if not os.path.exists(readme_path):
        return None
        
    with open(readme_path, 'r') as f:
        content = f.read()
        
    # Parse quantization settings if they exist
    if "quantization config" not in content.lower():
        return None
        
    config = {}
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('- '):
            try:
                key, value = line[2:].split(': ')
                # Convert string values to appropriate types
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                elif value.replace('.', '').isdigit():
                    value = float(value)
                config[key] = value
            except:
                continue
                
    return config

def jload(f, mode="r"):
    """
    Load a .json or .jsonl file into a dictionary or list of dicts.
    Supports both multiline JSON and JSONL with fallback to line-by-line parsing.
    """
    f = _make_r_io_base(f, mode)
    try:
        jdict = json.load(f)
    except json.JSONDecodeError:
        f.seek(0)  # Reset file pointer
        lines = f.readlines()
        jdict = []
        for line in lines:
            try:
                if line.strip():
                    jdict.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON line: {e}")
                continue
    finally:
        f.close()
    return jdict