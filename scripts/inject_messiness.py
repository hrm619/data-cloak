#!/usr/bin/env python3
"""Inject realistic messiness into the synthetic base dataset.

Reads:  test_data/synthetic_base_10m.parquet
Writes: test_data/synthetic_messy_10m.parquet

Corruption patterns and approximate target rates:
  - NULLs: 5% country, 2% email, 1% name
  - Exact duplicates: 2% of rows appended
  - Name collision (same name, diff email): 3% rows appended
  - Email collision (same email, diff name): 1% rows appended
  - Character encoding edge cases: 1% diacritics, 0.5% mixed script, 0.3% emoji
  - Email format variations: 2% unusual TLD, 1% subdomain, 0.5% plus, 0.5% numeric
  - Whitespace: 1% leading/trailing, 0.5% internal double-space, 0.3% mixed case
  - Name anomalies: 0.5% single-word, 0.3% very long, 0.2% numeric
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(__file__).parent.parent / "test_data" / "synthetic_base_10m.parquet"
OUTPUT_PATH = Path(__file__).parent.parent / "test_data" / "synthetic_messy_10m.parquet"

DIACRITIC_NAMES = [
    "José María", "François Dupont", "Müller Hans", "Søren Kierkegaard",
    "李明", "محمد العلي", "Владимир Иванов", "Ångström Lars",
    "Ñoño García", "Ünsal Öztürk",
]

MIXED_SCRIPT_NAMES = [
    "田中 Tanaka", "José 李", "Kim 김민준", "Иван Smith",
    "Ahmad أحمد", "Wang 王伟 Wei",
]

UNUSUAL_TLDS = [".museum", ".co.uk", ".co.jp", ".com.br", ".de", ".fr", ".io"]


def inject_nulls(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace a percentage of values with NaN in each PII column."""
    n = len(df)
    for col, rate in [("country", 0.05), ("email", 0.02), ("name", 0.01)]:
        idx = rng.choice(n, size=int(n * rate), replace=False)
        df.loc[idx, col] = np.nan
    return df


def inject_exact_duplicates(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Append 2% of rows as exact duplicates."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.02), replace=False)
    return pd.concat([df, df.iloc[idx]], ignore_index=True)


def inject_name_collisions(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Append 3% of rows with the same name but a different email."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.03), replace=False)
    copies = df.iloc[idx].copy().reset_index(drop=True)
    copies["email"] = copies["email"].apply(
        lambda e: f"alt_{rng.integers(1000, 9999)}@example.com" if pd.notna(e) else e
    )
    return pd.concat([df, copies], ignore_index=True)


def inject_email_collisions(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Append 1% of rows with the same email but a different name."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.01), replace=False)
    copies = df.iloc[idx].copy().reset_index(drop=True)
    copies["name"] = copies["name"].apply(
        lambda name: f"Alt User {rng.integers(100, 999)}" if pd.notna(name) else name
    )
    return pd.concat([df, copies], ignore_index=True)


def inject_diacritics(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 1% of names with diacritic/non-Latin characters."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[idx, "name"] = [random.choice(DIACRITIC_NAMES) for _ in range(len(idx))]
    return df


def inject_mixed_scripts(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 0.5% of names with mixed-script values."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df.loc[idx, "name"] = [random.choice(MIXED_SCRIPT_NAMES) for _ in range(len(idx))]
    return df


def inject_emoji_names(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 0.3% of names with emoji-containing strings."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.003), replace=False)
    emojis = ["😀", "🎉", "🌍", "🔥", "✨"]
    df.loc[idx, "name"] = df.loc[idx, "name"].apply(
        lambda name: f"{name}{random.choice(emojis)}" if pd.notna(name) else name
    )
    return df


def inject_unusual_tlds(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 2% of email domains with unusual TLDs."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[idx, "email"] = df.loc[idx, "email"].apply(
        lambda e: e.rsplit(".", 1)[0] + random.choice(UNUSUAL_TLDS) if pd.notna(e) and "." in e else e
    )
    return df


def inject_subdomains(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 1% of email domains with subdomain variants."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[idx, "email"] = df.loc[idx, "email"].apply(
        lambda e: e.replace("@", "@mail.") if pd.notna(e) else e
    )
    return df


def inject_plus_addressing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add plus-addressing to 0.5% of emails."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df.loc[idx, "email"] = df.loc[idx, "email"].apply(
        lambda e: e.replace("@", f"+tag{rng.integers(1, 99)}@") if pd.notna(e) else e
    )
    return df


def inject_numeric_emails(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 0.5% of email usernames with numeric-heavy strings."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df.loc[idx, "email"] = df.loc[idx, "email"].apply(
        lambda e: f"{rng.integers(100000, 999999)}@" + e.split("@")[1] if pd.notna(e) and "@" in e else e
    )
    return df


def inject_whitespace(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add leading/trailing whitespace to 1% of names and emails."""
    n = len(df)
    for col in ("name", "email"):
        idx = rng.choice(n, size=int(n * 0.01), replace=False)
        df.loc[idx, col] = df.loc[idx, col].apply(
            lambda v: f"  {v}  " if pd.notna(v) else v
        )
    return df


def inject_double_spaces(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Insert double internal spaces into 0.5% of names."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df.loc[idx, "name"] = df.loc[idx, "name"].apply(
        lambda v: v.replace(" ", "  ", 1) if pd.notna(v) and " " in v else v
    )
    return df


def inject_mixed_case(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply all-caps or all-lowercase to 0.3% of names."""
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.003), replace=False)
    df.loc[idx, "name"] = df.loc[idx, "name"].apply(
        lambda v: v.upper() if pd.notna(v) and random.random() < 0.5 else (v.lower() if pd.notna(v) else v)
    )
    return df


def inject_single_word_names(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 0.5% of names with single-word names."""
    single_words = ["Cher", "Prince", "Madonna", "Zendaya", "Adele", "Banksy"]
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df.loc[idx, "name"] = [random.choice(single_words) for _ in range(len(idx))]
    return df


def inject_long_names(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 0.3% of names with 5+ word names."""
    long_names = [
        "Mary Jane Watson Parker Smith",
        "Jean-Baptiste Emmanuel Zorg Le Grand",
        "Sir Arthur Ignatius Conan Doyle III",
    ]
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.003), replace=False)
    df.loc[idx, "name"] = [random.choice(long_names) for _ in range(len(idx))]
    return df


def inject_numeric_names(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace 0.2% of names with numeric-containing values."""
    numeric_names = ["John 2nd", "123 Main", "User 404", "Agent 007"]
    n = len(df)
    idx = rng.choice(n, size=int(n * 0.002), replace=False)
    df.loc[idx, "name"] = [random.choice(numeric_names) for _ in range(len(idx))]
    return df


def validate(df: pd.DataFrame, total_base: int) -> None:
    """Print a summary of the corrupted dataset for manual validation."""
    print(f"  Total rows:      {len(df):,}  (base was {total_base:,})")
    for col in ("name", "email", "country"):
        null_pct = df[col].isna().mean() * 100
        print(f"  NULL {col:<8}: {null_pct:.2f}%")


def main() -> None:
    print(f"Reading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    total_base = len(df)
    print(f"Loaded {total_base:,} rows.")

    rng = np.random.default_rng(seed=42)

    steps = [
        ("Injecting NULLs",             inject_nulls),
        ("Injecting exact duplicates",   inject_exact_duplicates),
        ("Injecting name collisions",    inject_name_collisions),
        ("Injecting email collisions",   inject_email_collisions),
        ("Injecting diacritics",         inject_diacritics),
        ("Injecting mixed scripts",      inject_mixed_scripts),
        ("Injecting emoji names",        inject_emoji_names),
        ("Injecting unusual TLDs",       inject_unusual_tlds),
        ("Injecting subdomains",         inject_subdomains),
        ("Injecting plus addressing",    inject_plus_addressing),
        ("Injecting numeric emails",     inject_numeric_emails),
        ("Injecting whitespace",         inject_whitespace),
        ("Injecting double spaces",      inject_double_spaces),
        ("Injecting mixed case",         inject_mixed_case),
        ("Injecting single-word names",  inject_single_word_names),
        ("Injecting long names",         inject_long_names),
        ("Injecting numeric names",      inject_numeric_names),
    ]

    for label, fn in steps:
        print(f"{label} ...")
        df = fn(df, rng)

    print("\nValidation:")
    validate(df, total_base)

    print(f"\nWriting {OUTPUT_PATH} ...")
    df.to_parquet(OUTPUT_PATH, compression="zstd", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
