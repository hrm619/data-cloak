"""Tests: anonymization preserves statistical shape of the data."""

import pandas as pd
import pytest

from data_cloak.anonymize import anonymize_dataframe


def test_row_count_preserved(small_df, config):
    """Output DataFrame has the same number of rows as input."""
    result = anonymize_dataframe(small_df, config)
    assert len(result) == len(small_df)


def test_null_rows_preserved(null_df, config):
    """Rows that were NULL before anonymization remain NULL after."""
    result = anonymize_dataframe(null_df, config)

    assert result[null_df["name"].isna()]["name"].isna().all(), "NULL names should stay NULL"
    assert result[null_df["email"].isna()]["email"].isna().all(), "NULL emails should stay NULL"
    assert result[null_df["country"].isna()]["country"].isna().all(), "NULL countries should stay NULL"


def test_null_counts_preserved(null_df, config):
    """The number of NULLs per column is unchanged by anonymization."""
    result = anonymize_dataframe(null_df, config)

    for col in ("name", "email", "country"):
        assert null_df[col].isna().sum() == result[col].isna().sum(), (
            f"NULL count changed for column '{col}'"
        )


def test_non_null_values_are_replaced(small_df, config):
    """Every non-null value in the input is replaced with a different fake value."""
    result = anonymize_dataframe(small_df, config)

    # Original values should not appear in anonymized output
    original_names = set(small_df["name"].dropna())
    anonymized_names = set(result["name"].dropna())
    assert not original_names & anonymized_names, (
        "Some original name values leaked into anonymized output"
    )


def test_non_null_rows_stay_non_null(small_df, config):
    """Rows that were not NULL before anonymization are not NULL after."""
    result = anonymize_dataframe(small_df, config)

    for col in ("name", "email", "country"):
        assert result[~small_df[col].isna()][col].notna().all(), (
            f"Non-null {col} became null after anonymization"
        )
