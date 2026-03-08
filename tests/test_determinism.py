"""Tests: same input always produces the same anonymized output."""

import pandas as pd

from data_cloak.anonymize import anonymize_dataframe, anonymize_value


def test_single_value_is_deterministic():
    """The same raw value always anonymizes to the same fake value."""
    assert anonymize_value("John Smith", "name") == anonymize_value("John Smith", "name")
    assert anonymize_value("john@example.com", "email") == anonymize_value("john@example.com", "email")
    assert anonymize_value("US", "country") == anonymize_value("US", "country")


def test_single_row_run_twice(config):
    """A single-row DataFrame produces identical output on two calls."""
    row = pd.DataFrame([{"name": "John Smith", "email": "john@example.com", "country": "US"}])
    result1 = anonymize_dataframe(row, config)
    result2 = anonymize_dataframe(row, config)
    pd.testing.assert_frame_equal(result1, result2)


def test_full_dataset_run_twice(small_df, config):
    """A 100-row DataFrame produces byte-identical output on two calls."""
    result1 = anonymize_dataframe(small_df, config)
    result2 = anonymize_dataframe(small_df, config)
    pd.testing.assert_frame_equal(result1, result2)


def test_column_order_does_not_affect_output(config):
    """Anonymization result is independent of column order in the DataFrame."""
    df1 = pd.DataFrame([{"name": "Alice", "email": "alice@example.com", "country": "DE"}])
    df2 = df1[["country", "email", "name"]]
    r1 = anonymize_dataframe(df1, config)
    r2 = anonymize_dataframe(df2, config)
    assert r1["name"].iloc[0] == r2["name"].iloc[0]
    assert r1["email"].iloc[0] == r2["email"].iloc[0]
    assert r1["country"].iloc[0] == r2["country"].iloc[0]


def test_distinct_inputs_can_produce_distinct_outputs():
    """Different input values should not always hash to the same fake value."""
    names = [f"Person {i}" for i in range(50)]
    results = {anonymize_value(n, "name") for n in names}
    assert len(results) > 1, "All 50 distinct names collapsed to a single anonymized value"
