from pathlib import Path

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Args:
        path: Path to the input CSV file.

    Returns:
        DataFrame containing the CSV contents.
    """
    return pd.read_csv(path, encoding_errors="replace", encoding="utf-8-sig", index_col=False)


def anon_path(path: str | Path) -> Path:
    """Derive the anonymized output path from an input CSV path.

    Appends '_ANON' to the stem of the filename, preserving the suffix and
    parent directory. For example, 'data/users.csv' -> 'data/users_ANON.csv'.

    Args:
        path: Path to the original CSV file.

    Returns:
        Path with '_ANON' appended to the filename stem.
    """
    p = Path(path)
    return p.with_name(f"{p.stem}_ANON{p.suffix}")


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to a CSV file.

    Args:
        df: DataFrame to write.
        path: Destination file path.
    """
    pd.DataFrame.to_csv(df, path, index=False)
