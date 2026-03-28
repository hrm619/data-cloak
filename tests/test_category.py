"""Tests for category field type anonymization."""

import pandas as pd

from data_cloak.anonymize import (
    anonymize_category,
    anonymize_column,
    anonymize_dataframe,
    anonymize_value,
    _CATEGORIES,
)


class TestAnonymizeCategory:

    def test_returns_string(self):
        result = anonymize_category("small")
        assert isinstance(result, str) and len(result) > 0

    def test_deterministic(self):
        assert anonymize_category("small") == anonymize_category("small")

    def test_result_from_pool(self):
        assert anonymize_category("medium") in _CATEGORIES

    def test_different_inputs_can_differ(self):
        results = {anonymize_category(v) for v in ["small", "medium", "large", "XL", "XXL"]}
        assert len(results) > 1

    def test_case_sensitive(self):
        assert anonymize_category("Small") != anonymize_category("small")

    def test_abbreviations(self):
        for abbr in ("S", "M", "L", "XL"):
            result = anonymize_category(abbr)
            assert result in _CATEGORIES


class TestCategoryViaDispatcher:

    def test_anonymize_value(self):
        result = anonymize_value("High", "category")
        assert result in _CATEGORIES

    def test_unsupported_type_still_raises(self):
        import pytest
        with pytest.raises(ValueError):
            anonymize_value("x", "bogus_type")


class TestCategoryColumn:

    def test_anonymize_column(self):
        col = pd.Series(["small", "medium", "large", "small"])
        result = anonymize_column(col, "category")
        assert len(result) == 4
        assert all(v in _CATEGORIES for v in result)

    def test_preserves_nulls(self):
        col = pd.Series(["small", None, "large"])
        result = anonymize_column(col, "category")
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.notna(result.iloc[2])

    def test_deterministic_column(self):
        col = pd.Series(["S", "M", "L", "S"])
        r1 = anonymize_column(col, "category")
        r2 = anonymize_column(col, "category")
        pd.testing.assert_series_equal(r1, r2)

    def test_same_input_maps_to_same_output(self):
        col = pd.Series(["small", "medium", "small", "large", "medium"])
        result = anonymize_column(col, "category")
        assert result.iloc[0] == result.iloc[2]
        assert result.iloc[1] == result.iloc[4]


class TestCategoryDataframe:

    def test_full_pipeline(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "size": ["small", "large"],
        })
        config = {"name": "name", "size": "category"}
        result = anonymize_dataframe(df, config)
        assert all(v in _CATEGORIES for v in result["size"])

    def test_category_column_not_affected_by_filename(self):
        df = pd.DataFrame({"size": ["S", "M", "L"]})
        config = {"size": "category"}
        r1 = anonymize_dataframe(df, config, filename="a.csv")
        r2 = anonymize_dataframe(df, config, filename="b.csv")
        pd.testing.assert_series_equal(r1["size"], r2["size"])
