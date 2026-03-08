"""Tests: NULL values pass through correctly and don't corrupt neighboring rows."""

import pandas as pd
import pytest

from data_cloak.anonymize import anonymize_dataframe

CONFIG = {"name": "name", "email": "email", "country": "country"}


def test_null_name_passes_through():
    """A NULL name stays NULL; the adjacent email row is unaffected."""
    df = pd.DataFrame([
        {"name": None,   "email": "test@example.com", "country": "US"},
        {"name": "John", "email": "john@example.com", "country": "US"},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert pd.isna(result.iloc[0]["name"]), "NULL name should remain NULL"
    assert pd.notna(result.iloc[1]["name"]), "Non-null name should be anonymized, not nullified"


def test_null_email_passes_through():
    """A NULL email stays NULL."""
    df = pd.DataFrame([
        {"name": "John", "email": None, "country": "US"},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert pd.isna(result.iloc[0]["email"]), "NULL email should remain NULL"


def test_null_country_passes_through():
    """A NULL country stays NULL."""
    df = pd.DataFrame([
        {"name": "John", "email": "john@example.com", "country": None},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert pd.isna(result.iloc[0]["country"]), "NULL country should remain NULL"


def test_all_columns_null_row():
    """A fully-NULL row should not crash and should remain fully NULL."""
    df = pd.DataFrame([
        {"name": None, "email": None, "country": None},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert len(result) == 1, "Row count should be preserved"
    assert result.iloc[0].isna().all(), "All-NULL row should remain all-NULL"


def test_null_does_not_contaminate_adjacent_rows(null_df, config):
    """NULLs in one row must not affect the anonymized values of other rows."""
    result = anonymize_dataframe(null_df, config)

    # Row 4 (index 4): Alice with clean data — should be fully non-null
    assert pd.notna(result.iloc[4]["name"])
    assert pd.notna(result.iloc[4]["email"])
    assert pd.notna(result.iloc[4]["country"])


@pytest.mark.parametrize("col", ["name", "email", "country"])
def test_partial_null_column(col):
    """Mixed NULL/non-null column: NULLs stay NULL, non-nulls get anonymized."""
    df = pd.DataFrame([
        {"name": "Alice", "email": "alice@example.com", "country": "US"},
        {"name": None,    "email": None,                 "country": None},
        {"name": "Bob",   "email": "bob@example.com",   "country": "DE"},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert pd.isna(result.iloc[1][col]), f"NULL {col} at row 1 should remain NULL"
    assert pd.notna(result.iloc[0][col]), f"Non-null {col} at row 0 should be anonymized"
    assert pd.notna(result.iloc[2][col]), f"Non-null {col} at row 2 should be anonymized"
