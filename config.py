import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict[str, str | dict]:
    """Load column-to-field-type mappings from a TOML config file.

    Args:
        path: Path to a TOML file containing a [columns] table.

    Returns:
        Dict mapping column name -> field_type string or dict
        (e.g. {"Amount": "amount", "Description": {"type": "description", "category_column": "Category"}}).

    Raises:
        KeyError: if the [columns] table is missing from the file.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if "columns" not in data:
        raise KeyError(f"Missing [columns] table in {path}")
    return data["columns"]
