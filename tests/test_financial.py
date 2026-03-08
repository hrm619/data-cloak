"""Tests for financial extension: date shifting, amount scaling, and description anonymization."""

import pandas as pd
import pytest

from data_cloak.anonymize import (
    anonymize_amount,
    anonymize_dataframe,
    anonymize_date,
    anonymize_description,
    anonymize_description_column,
    anonymize_value,
    derive_multiplier,
    derive_offset,
)


# --- derive_offset / derive_multiplier ---

class TestDeriveConstants:

    def test_offset_is_deterministic(self):
        assert derive_offset("data.csv") == derive_offset("data.csv")

    def test_multiplier_is_deterministic(self):
        assert derive_multiplier("data.csv") == derive_multiplier("data.csv")

    def test_offset_in_range(self):
        for name in ("a.csv", "b.csv", "transactions_2025.csv", "big_file.parquet"):
            offset = derive_offset(name)
            assert 180 <= offset <= 730, f"offset {offset} out of range for {name}"

    def test_multiplier_in_range(self):
        for name in ("a.csv", "b.csv", "transactions_2025.csv", "big_file.parquet"):
            mult = derive_multiplier(name)
            assert 1.5 <= mult <= 3.5, f"multiplier {mult} out of range for {name}"

    def test_different_filenames_can_produce_different_values(self):
        offsets = {derive_offset(f"file_{i}.csv") for i in range(20)}
        assert len(offsets) > 1, "All filenames produced the same offset"


# --- anonymize_date ---

class TestAnonymizeDate:

    def test_shifts_forward(self):
        result = anonymize_date("01/01/2025", 30)
        assert result == "01/31/2025"

    def test_preserves_mmddyyyy_format(self):
        result = anonymize_date("03/15/2024", 100)
        parts = result.split("/")
        assert len(parts) == 3
        assert len(parts[0]) == 2 and len(parts[1]) == 2 and len(parts[2]) == 4

    def test_handles_month_rollover(self):
        result = anonymize_date("12/15/2024", 30)
        assert result == "01/14/2025"

    def test_handles_leap_year(self):
        result = anonymize_date("02/28/2024", 1)
        assert result == "02/29/2024"  # 2024 is a leap year

    def test_deterministic(self):
        assert anonymize_date("06/15/2024", 200) == anonymize_date("06/15/2024", 200)

    def test_via_anonymize_value(self):
        result = anonymize_value("01/01/2025", "date", offset=10)
        assert result == "01/11/2025"


# --- anonymize_amount ---

class TestAnonymizeAmount:

    def test_scales_positive(self):
        assert anonymize_amount("100.00", 2.0) == 200.0

    def test_preserves_negative_sign(self):
        result = anonymize_amount("-50.00", 2.0)
        assert result < 0, "Negative amount should stay negative"
        assert result == -100.0

    def test_rounds_to_two_decimals(self):
        result = anonymize_amount("33.33", 1.111)
        assert result == round(33.33 * 1.111, 2)

    def test_accepts_numeric_input(self):
        assert anonymize_amount(100, 2.5) == 250.0

    def test_deterministic(self):
        assert anonymize_amount("42.50", 1.75) == anonymize_amount("42.50", 1.75)

    def test_via_anonymize_value(self):
        result = anonymize_value("100", "amount", multiplier=2.0)
        assert result == 200.0


# --- anonymize_description ---

class TestAnonymizeDescription:

    def test_returns_string(self):
        result = anonymize_description("STARBUCKS #1234", "Food & Drink")
        assert isinstance(result, str) and len(result) > 0

    def test_deterministic(self):
        a = anonymize_description("STARBUCKS #1234", "Food & Drink")
        b = anonymize_description("STARBUCKS #1234", "Food & Drink")
        assert a == b

    def test_different_inputs_can_differ(self):
        results = {anonymize_description(f"MERCHANT_{i}", "Shopping") for i in range(20)}
        assert len(results) > 1

    def test_falls_back_to_default_category(self):
        result = anonymize_description("UNKNOWN VENDOR", "Nonexistent Category")
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.parametrize("category", [
        "Travel", "Bills & Utilities", "Professional Services", "Food & Drink",
        "Shopping", "Health & Wellness", "Entertainment", "ATM/Cash", "Transfer",
    ])
    def test_all_categories_return_valid_result(self, category):
        result = anonymize_description("TEST MERCHANT", category)
        assert isinstance(result, str) and len(result) > 0

    def test_via_anonymize_value(self):
        result = anonymize_value("STARBUCKS", "description", category="Food & Drink")
        assert isinstance(result, str)


# --- anonymize_description_column ---

class TestAnonymizeDescriptionColumn:

    def test_maps_descriptions_by_category(self):
        descs = pd.Series(["DELTA AIR", "COMCAST BILL", "STARBUCKS"])
        cats = pd.Series(["Travel", "Bills & Utilities", "Food & Drink"])
        result = anonymize_description_column(descs, cats)
        assert len(result) == 3
        assert result.notna().all()

    def test_preserves_nulls(self):
        descs = pd.Series(["DELTA AIR", None, "STARBUCKS"])
        cats = pd.Series(["Travel", "Bills & Utilities", "Food & Drink"])
        result = anonymize_description_column(descs, cats)
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.notna(result.iloc[2])


# --- anonymize_dataframe with financial config ---

class TestFinancialDataframe:

    @pytest.fixture
    def credit_card_df(self):
        return pd.DataFrame({
            "Transaction Date": ["01/15/2025", "02/20/2025", "03/10/2025"],
            "Amount": ["125.50", "-42.00", "8.99"],
            "Description": ["DELTA AIRLINES", "REFUND - AMAZON", "STARBUCKS"],
            "Category": ["Travel", "Shopping", "Food & Drink"],
        })

    @pytest.fixture
    def financial_config(self):
        return {
            "Transaction Date": "date",
            "Amount": "amount",
            "Description": {"type": "description", "category_column": "Category"},
        }

    def test_full_pipeline(self, credit_card_df, financial_config):
        result = anonymize_dataframe(credit_card_df, financial_config, filename="test.csv")
        assert len(result) == 3

    def test_dates_are_shifted(self, credit_card_df, financial_config):
        result = anonymize_dataframe(credit_card_df, financial_config, filename="test.csv")
        # All dates should differ from originals
        for i in range(3):
            assert result.iloc[i]["Transaction Date"] != credit_card_df.iloc[i]["Transaction Date"]

    def test_amounts_are_scaled(self, credit_card_df, financial_config):
        result = anonymize_dataframe(credit_card_df, financial_config, filename="test.csv")
        # Negative amount should remain negative
        assert float(result.iloc[1]["Amount"]) < 0

    def test_descriptions_are_replaced(self, credit_card_df, financial_config):
        result = anonymize_dataframe(credit_card_df, financial_config, filename="test.csv")
        for i in range(3):
            assert result.iloc[i]["Description"] != credit_card_df.iloc[i]["Description"]

    def test_deterministic_with_same_filename(self, credit_card_df, financial_config):
        r1 = anonymize_dataframe(credit_card_df, financial_config, filename="test.csv")
        r2 = anonymize_dataframe(credit_card_df, financial_config, filename="test.csv")
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_filenames_produce_different_offsets(self, credit_card_df, financial_config):
        r1 = anonymize_dataframe(credit_card_df, financial_config, filename="file_a.csv")
        r2 = anonymize_dataframe(credit_card_df, financial_config, filename="file_b.csv")
        # Dates should differ (different offset derived from different filenames)
        assert r1.iloc[0]["Transaction Date"] != r2.iloc[0]["Transaction Date"]

    def test_category_column_not_modified(self, credit_card_df, financial_config):
        result = anonymize_dataframe(credit_card_df, financial_config, filename="test.csv")
        pd.testing.assert_series_equal(result["Category"], credit_card_df["Category"])

    def test_null_handling_in_financial_columns(self, financial_config):
        df = pd.DataFrame({
            "Transaction Date": [None, "01/15/2025"],
            "Amount": ["100.00", None],
            "Description": [None, "STARBUCKS"],
            "Category": ["Travel", "Food & Drink"],
        })
        result = anonymize_dataframe(df, financial_config, filename="test.csv")
        assert pd.isna(result.iloc[0]["Transaction Date"])
        assert pd.isna(result.iloc[1]["Amount"])
        assert pd.isna(result.iloc[0]["Description"])
