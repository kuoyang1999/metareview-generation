import io
import json
import logging

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

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