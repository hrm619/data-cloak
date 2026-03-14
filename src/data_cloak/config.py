import tomllib
from pathlib import Path

import yaml


def _load_toml(path: Path) -> dict:
    """Read a TOML file and return the parsed dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_yaml(path: Path) -> dict:
    """Read a YAML file and return the parsed dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_columns(data: dict, path: Path) -> dict[str, str | dict]:
    """Extract and normalize the columns table from parsed config data.

    Args:
        data: Parsed config dict.
        path: Original file path (for error messages).

    Returns:
        Dict mapping column name -> field_type string or dict.

    Raises:
        KeyError: if the [columns] table is missing.
    """
    if "columns" not in data:
        raise KeyError(f"Missing [columns] table in {path}")
    columns = data["columns"]
    return {
        col: entry["type"] if isinstance(entry, dict) and "confidence" in entry else entry
        for col, entry in columns.items()
    }


def load_config(path: str | Path) -> dict[str, str | dict]:
    """Load column-to-field-type mappings from a TOML or YAML config file.

    Detects format by file extension (.yaml/.yml → YAML, otherwise TOML).

    Args:
        path: Path to a config file containing a [columns] table.

    Returns:
        Dict mapping column name -> field_type string or dict
        (e.g. {"Amount": "amount", "Description": {"type": "description", "category_column": "Category"}}).

    Raises:
        KeyError: if the [columns] table is missing from the file.
    """
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        data = _load_yaml(path)
    else:
        data = _load_toml(path)
    return _extract_columns(data, path)
