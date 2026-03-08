"""Tests: correct behavior when multiple rows share the same PII values."""

import pandas as pd

from anonymize import anonymize_dataframe

CONFIG = {"name": "name", "email": "email", "country": "country"}


def test_exact_duplicate_rows_anonymize_identically():
    """Two identical rows must produce identical anonymized output."""
    df = pd.DataFrame([
        {"name": "John Smith", "email": "john@example.com", "country": "US"},
        {"name": "John Smith", "email": "john@example.com", "country": "US"},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert result.iloc[0]["name"] == result.iloc[1]["name"], "Duplicate names should hash identically"
    assert result.iloc[0]["email"] == result.iloc[1]["email"], "Duplicate emails should hash identically"
    assert result.iloc[0]["country"] == result.iloc[1]["country"], "Duplicate countries should hash identically"


def test_same_name_different_emails():
    """Same name always maps to same anonymized name; different emails map differently."""
    df = pd.DataFrame([
        {"name": "John Smith", "email": "john1@example.com", "country": "US"},
        {"name": "John Smith", "email": "john2@example.com", "country": "US"},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert result.iloc[0]["name"] == result.iloc[1]["name"], (
        "Same input name should always produce the same anonymized name"
    )
    assert result.iloc[0]["email"] != result.iloc[1]["email"], (
        "Different input emails should (almost certainly) produce different anonymized emails"
    )


def test_same_email_different_names():
    """Same email always maps to same anonymized email; different names map differently."""
    df = pd.DataFrame([
        {"name": "John Smith",  "email": "contact@example.com", "country": "US"},
        {"name": "Jane Smith",  "email": "contact@example.com", "country": "US"},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert result.iloc[0]["email"] == result.iloc[1]["email"], (
        "Same input email should always produce the same anonymized email"
    )
    assert result.iloc[0]["name"] != result.iloc[1]["name"], (
        "Different input names should (almost certainly) produce different anonymized names"
    )


def test_collision_df_row_count_preserved(collision_df, config):
    """Collision fixture passes through with the same number of rows."""
    result = anonymize_dataframe(collision_df, config)
    assert len(result) == len(collision_df)
