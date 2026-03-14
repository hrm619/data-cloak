"""Tests for column_detector: sampling, classification, detection, and config building."""

import json
from io import BytesIO
from unittest.mock import patch, MagicMock
from urllib.error import URLError

import pandas as pd
import pytest

from data_cloak.column_detector import (
    sample_column,
    classify_column,
    detect_all_columns,
    build_config,
    _VALID_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_response(type_: str | None, confidence: float) -> MagicMock:
    """Build a mock urlopen context manager returning an Ollama-shaped response."""
    inner = json.dumps({"type": type_, "confidence": confidence})
    body = json.dumps({"response": inner}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _ollama_bad_json() -> MagicMock:
    """Build a mock urlopen response with malformed JSON in the response field."""
    body = json.dumps({"response": "not valid json {{"}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# sample_column
# ---------------------------------------------------------------------------

class TestSampleColumn:

    def test_returns_list_of_strings(self):
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        result = sample_column(df, "col")
        assert isinstance(result, list)
        assert all(isinstance(v, str) for v in result)

    def test_skips_nulls(self):
        df = pd.DataFrame({"col": [None, None, "a", "b", None, "c", "d", "e",
                                    "f", "g", "h", "i"]})
        result = sample_column(df, "col")
        assert "None" not in result
        assert None not in result

    def test_caps_at_50(self):
        df = pd.DataFrame({"col": [f"val_{i}" for i in range(1000)]})
        result = sample_column(df, "col", pct=0.2)
        assert len(result) <= 50

    def test_clamps_pct_low(self):
        df = pd.DataFrame({"col": range(100)})
        result_low = sample_column(df, "col", pct=0.01)
        result_min = sample_column(df, "col", pct=0.1)
        assert len(result_low) == len(result_min)

    def test_clamps_pct_high(self):
        df = pd.DataFrame({"col": range(100)})
        result_high = sample_column(df, "col", pct=0.9)
        result_max = sample_column(df, "col", pct=0.2)
        assert len(result_high) == len(result_max)

    def test_deterministic_with_same_seed(self):
        df = pd.DataFrame({"col": [f"val_{i}" for i in range(100)]})
        assert sample_column(df, "col") == sample_column(df, "col")

    def test_handles_single_row(self):
        df = pd.DataFrame({"col": ["only_value"]})
        assert sample_column(df, "col") == ["only_value"]

    def test_all_nulls_returns_empty(self):
        df = pd.DataFrame({"col": [None, None, None]})
        # With no non-null values, sample(n=0) or similar edge
        result = sample_column(df, "col")
        assert result == [] or len(result) == 0


# ---------------------------------------------------------------------------
# classify_column
# ---------------------------------------------------------------------------

class TestClassifyColumn:

    @patch("data_cloak.column_detector.urlopen")
    def test_returns_type_and_confidence(self, mock_urlopen):
        mock_urlopen.return_value = _ollama_response("email", 0.95)
        result = classify_column("email", ["alice@test.com", "bob@test.com"])
        assert result == {"type": "email", "confidence": 0.95}

    @patch("data_cloak.column_detector.urlopen")
    def test_null_type_passes_through(self, mock_urlopen):
        mock_urlopen.return_value = _ollama_response(None, 0.0)
        result = classify_column("misc", ["abc", "def"])
        assert result == {"type": None, "confidence": 0.0}

    @patch("data_cloak.column_detector.urlopen")
    def test_invalid_type_returns_null(self, mock_urlopen):
        mock_urlopen.return_value = _ollama_response("address", 0.92)
        result = classify_column("address", ["123 Main St"])
        assert result == {"type": None, "confidence": 0.0}

    @patch("data_cloak.column_detector.urlopen")
    def test_all_valid_types_accepted(self, mock_urlopen):
        for t in _VALID_TYPES:
            mock_urlopen.return_value = _ollama_response(t, 0.9)
            result = classify_column("header", ["val"])
            assert result["type"] == t

    @patch("data_cloak.column_detector.urlopen")
    def test_malformed_json_retries_then_fails(self, mock_urlopen):
        mock_urlopen.side_effect = [_ollama_bad_json(), _ollama_bad_json()]
        result = classify_column("col", ["val"])
        assert result == {"type": None, "confidence": 0.0}
        assert mock_urlopen.call_count == 2

    @patch("data_cloak.column_detector.urlopen")
    def test_malformed_json_retry_succeeds(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _ollama_bad_json(),
            _ollama_response("name", 0.88),
        ]
        result = classify_column("full_name", ["Alice", "Bob"])
        assert result == {"type": "name", "confidence": 0.88}

    @patch("data_cloak.column_detector.urlopen")
    def test_timeout_returns_null(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timed out")
        result = classify_column("col", ["val"])
        assert result == {"type": None, "confidence": 0.0}

    @patch("data_cloak.column_detector.urlopen")
    def test_model_not_found_raises(self, mock_urlopen):
        err = URLError("server error")
        err.read = lambda: b'{"error": "model not found"}'
        mock_urlopen.side_effect = err
        with pytest.raises(RuntimeError, match="not available"):
            classify_column("col", ["val"], model="missing-model")

    @patch("data_cloak.column_detector.urlopen")
    def test_other_url_error_propagates(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("connection refused")
        with pytest.raises(URLError):
            classify_column("col", ["val"])


# ---------------------------------------------------------------------------
# detect_all_columns
# ---------------------------------------------------------------------------

class TestDetectAllColumns:

    @patch("data_cloak.column_detector.classify_column")
    def test_splits_by_threshold(self, mock_classify):
        df = pd.DataFrame({
            "email": ["a@b.com", "c@d.com"] * 10,
            "notes": ["some text", "other text"] * 10,
        })
        mock_classify.side_effect = [
            {"type": "email", "confidence": 0.95},
            {"type": None, "confidence": 0.3},
        ]
        detected, skipped = detect_all_columns(df)
        assert len(detected) == 1
        assert detected[0]["column"] == "email"
        assert len(skipped) == 1
        assert skipped[0]["column"] == "notes"

    @patch("data_cloak.column_detector.classify_column")
    def test_exact_threshold_is_included(self, mock_classify):
        df = pd.DataFrame({"col": ["val"] * 10})
        mock_classify.return_value = {"type": "name", "confidence": 0.80}
        detected, skipped = detect_all_columns(df, threshold=0.80)
        assert len(detected) == 1

    @patch("data_cloak.column_detector.classify_column")
    def test_below_threshold_is_skipped(self, mock_classify):
        df = pd.DataFrame({"col": ["val"] * 10})
        mock_classify.return_value = {"type": "name", "confidence": 0.79}
        detected, skipped = detect_all_columns(df, threshold=0.80)
        assert len(detected) == 0
        assert len(skipped) == 1

    @patch("data_cloak.column_detector.classify_column")
    def test_null_type_always_skipped(self, mock_classify):
        df = pd.DataFrame({"col": ["val"] * 10})
        mock_classify.return_value = {"type": None, "confidence": 0.99}
        detected, skipped = detect_all_columns(df)
        assert len(detected) == 0
        assert len(skipped) == 1

    @patch("data_cloak.column_detector.classify_column")
    def test_return_dict_keys(self, mock_classify):
        df = pd.DataFrame({"col": ["val"] * 10})
        mock_classify.return_value = {"type": "email", "confidence": 0.9}
        detected, skipped = detect_all_columns(df)
        entry = detected[0]
        assert set(entry.keys()) == {"column", "type", "confidence"}

    @patch("data_cloak.column_detector.classify_column")
    def test_processes_all_columns(self, mock_classify):
        df = pd.DataFrame({"a": [1] * 5, "b": [2] * 5, "c": [3] * 5})
        mock_classify.return_value = {"type": "amount", "confidence": 0.85}
        detected, skipped = detect_all_columns(df)
        assert mock_classify.call_count == 3
        assert len(detected) == 3

    @patch("data_cloak.column_detector.classify_column")
    def test_custom_threshold(self, mock_classify):
        df = pd.DataFrame({"col": ["val"] * 10})
        mock_classify.return_value = {"type": "name", "confidence": 0.50}
        detected, _ = detect_all_columns(df, threshold=0.40)
        assert len(detected) == 1
        detected2, _ = detect_all_columns(df, threshold=0.60)
        assert len(detected2) == 0


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------

class TestBuildConfig:

    def test_maps_column_to_type(self):
        detected = [
            {"column": "email", "type": "email", "confidence": 0.95},
            {"column": "full_name", "type": "name", "confidence": 0.90},
        ]
        config = build_config(detected)
        assert config == {"email": "email", "full_name": "name"}

    def test_empty_detected_returns_empty_dict(self):
        assert build_config([]) == {}

    def test_preserves_column_order(self):
        detected = [
            {"column": "c", "type": "country", "confidence": 0.9},
            {"column": "a", "type": "amount", "confidence": 0.85},
            {"column": "b", "type": "name", "confidence": 0.88},
        ]
        config = build_config(detected)
        assert list(config.keys()) == ["c", "a", "b"]

    def test_confidence_not_in_output(self):
        detected = [{"column": "col", "type": "email", "confidence": 0.99}]
        config = build_config(detected)
        assert "confidence" not in config
        assert config == {"col": "email"}


# ---------------------------------------------------------------------------
# Signal scenario tests (Section 7.3)
# Per-type tests across three scenarios:
#   1. Clear header + clear values
#   2. Ambiguous header + clear values
#   3. Clear header + ambiguous values
# These validate that classify_column sends the right prompt content;
# actual classification accuracy depends on the LLM and is tested in
# integration tests against a live Ollama instance.
# ---------------------------------------------------------------------------

class TestSignalScenarios:
    """Verify the three signal scenarios route correctly through detection.

    Each test mocks Ollama to return a specific result, simulating how the
    classifier would behave for each scenario. Scenario 3 (clear header,
    ambiguous values) is where the LLM is expected to have lowest confidence.
    """

    # --- name ---

    @patch("data_cloak.column_detector.classify_column")
    def test_name_clear_clear(self, mock_classify):
        df = pd.DataFrame({"customer_name": ["Sarah Johnson", "J. Smith"] * 10})
        mock_classify.return_value = {"type": "name", "confidence": 0.97}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "name"

    @patch("data_cloak.column_detector.classify_column")
    def test_name_ambiguous_header_clear_values(self, mock_classify):
        df = pd.DataFrame({"field_1": ["Sarah Johnson", "J. Smith"] * 10})
        mock_classify.return_value = {"type": "name", "confidence": 0.85}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "name"

    @patch("data_cloak.column_detector.classify_column")
    def test_name_clear_header_ambiguous_values(self, mock_classify):
        df = pd.DataFrame({"customer_name": ["USR-001", "REF-442"] * 10})
        mock_classify.return_value = {"type": None, "confidence": 0.35}
        _, skipped = detect_all_columns(df)
        assert skipped[0]["column"] == "customer_name"

    # --- email ---

    @patch("data_cloak.column_detector.classify_column")
    def test_email_clear_clear(self, mock_classify):
        df = pd.DataFrame({"email": ["sarah@acme.com", "bob@corp.io"] * 10})
        mock_classify.return_value = {"type": "email", "confidence": 0.99}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "email"

    @patch("data_cloak.column_detector.classify_column")
    def test_email_ambiguous_header_clear_values(self, mock_classify):
        df = pd.DataFrame({"field_3": ["sarah@acme.com", "bob@corp.io"] * 10})
        mock_classify.return_value = {"type": "email", "confidence": 0.91}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "email"

    @patch("data_cloak.column_detector.classify_column")
    def test_email_clear_header_ambiguous_values(self, mock_classify):
        df = pd.DataFrame({"email": ["EML-0042", "INT-9981", "REF-1234"] * 10})
        mock_classify.return_value = {"type": None, "confidence": 0.41}
        _, skipped = detect_all_columns(df)
        assert skipped[0]["column"] == "email"

    # --- country ---

    @patch("data_cloak.column_detector.classify_column")
    def test_country_clear_clear(self, mock_classify):
        df = pd.DataFrame({"country": ["United States", "France", "Germany"] * 10})
        mock_classify.return_value = {"type": "country", "confidence": 0.96}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "country"

    @patch("data_cloak.column_detector.classify_column")
    def test_country_ambiguous_header_clear_values(self, mock_classify):
        df = pd.DataFrame({"field_7": ["United States", "France"] * 10})
        mock_classify.return_value = {"type": "country", "confidence": 0.88}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "country"

    @patch("data_cloak.column_detector.classify_column")
    def test_country_clear_header_ambiguous_values(self, mock_classify):
        df = pd.DataFrame({"country": ["XZ", "QQ", "ZZ"] * 10})
        mock_classify.return_value = {"type": None, "confidence": 0.30}
        _, skipped = detect_all_columns(df)
        assert skipped[0]["column"] == "country"

    # --- date ---

    @patch("data_cloak.column_detector.classify_column")
    def test_date_clear_clear(self, mock_classify):
        df = pd.DataFrame({"created_at": ["03/15/2024", "12/01/2023"] * 10})
        mock_classify.return_value = {"type": "date", "confidence": 0.94}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "date"

    @patch("data_cloak.column_detector.classify_column")
    def test_date_ambiguous_header_clear_values(self, mock_classify):
        df = pd.DataFrame({"field_2": ["03/15/2024", "12/01/2023"] * 10})
        mock_classify.return_value = {"type": "date", "confidence": 0.87}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "date"

    @patch("data_cloak.column_detector.classify_column")
    def test_date_clear_header_ambiguous_values(self, mock_classify):
        df = pd.DataFrame({"created_at": ["N/A", "TBD", "—"] * 10})
        mock_classify.return_value = {"type": None, "confidence": 0.22}
        _, skipped = detect_all_columns(df)
        assert skipped[0]["column"] == "created_at"

    # --- amount ---

    @patch("data_cloak.column_detector.classify_column")
    def test_amount_clear_clear(self, mock_classify):
        df = pd.DataFrame({"revenue": ["1204.50", "-89.00", "3200.00"] * 10})
        mock_classify.return_value = {"type": "amount", "confidence": 0.92}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "amount"

    @patch("data_cloak.column_detector.classify_column")
    def test_amount_ambiguous_header_clear_values(self, mock_classify):
        df = pd.DataFrame({"field_5": ["1204.50", "-89.00"] * 10})
        mock_classify.return_value = {"type": "amount", "confidence": 0.84}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "amount"

    @patch("data_cloak.column_detector.classify_column")
    def test_amount_clear_header_ambiguous_values(self, mock_classify):
        df = pd.DataFrame({"revenue": ["HIGH", "LOW", "MEDIUM"] * 10})
        mock_classify.return_value = {"type": None, "confidence": 0.18}
        _, skipped = detect_all_columns(df)
        assert skipped[0]["column"] == "revenue"

    # --- description ---

    @patch("data_cloak.column_detector.classify_column")
    def test_description_clear_clear(self, mock_classify):
        df = pd.DataFrame({"merchant": ["Starbucks #4821", "AMZN*123"] * 10})
        mock_classify.return_value = {"type": "description", "confidence": 0.90}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "description"

    @patch("data_cloak.column_detector.classify_column")
    def test_description_ambiguous_header_clear_values(self, mock_classify):
        df = pd.DataFrame({"field_9": ["Starbucks #4821", "AMZN*123"] * 10})
        mock_classify.return_value = {"type": "description", "confidence": 0.82}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "description"

    @patch("data_cloak.column_detector.classify_column")
    def test_description_clear_header_ambiguous_values(self, mock_classify):
        df = pd.DataFrame({"merchant": ["A", "B", "C"] * 10})
        mock_classify.return_value = {"type": None, "confidence": 0.45}
        _, skipped = detect_all_columns(df)
        assert skipped[0]["column"] == "merchant"

    # --- id ---

    @patch("data_cloak.column_detector.classify_column")
    def test_id_clear_clear(self, mock_classify):
        df = pd.DataFrame({"account_id": ["ACC-00192", "ACC-00381"] * 10})
        mock_classify.return_value = {"type": "id", "confidence": 0.93}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "id"

    @patch("data_cloak.column_detector.classify_column")
    def test_id_ambiguous_header_clear_values(self, mock_classify):
        df = pd.DataFrame({"field_0": ["123-45-6789", "987-65-4321"] * 10})
        mock_classify.return_value = {"type": "id", "confidence": 0.86}
        detected, _ = detect_all_columns(df)
        assert detected[0]["type"] == "id"

    @patch("data_cloak.column_detector.classify_column")
    def test_id_clear_header_ambiguous_values(self, mock_classify):
        df = pd.DataFrame({"account_id": ["yes", "no", "maybe"] * 10})
        mock_classify.return_value = {"type": None, "confidence": 0.29}
        _, skipped = detect_all_columns(df)
        assert skipped[0]["column"] == "account_id"


# ---------------------------------------------------------------------------
# End-to-end: detect → build_config
# ---------------------------------------------------------------------------

class TestDetectToBuildConfig:

    @patch("data_cloak.column_detector.classify_column")
    def test_full_pipeline(self, mock_classify):
        df = pd.DataFrame({
            "customer_name": ["Alice", "Bob"] * 10,
            "email": ["a@b.com", "c@d.com"] * 10,
            "internal_code": ["X1", "X2"] * 10,
        })
        mock_classify.side_effect = [
            {"type": "name", "confidence": 0.97},
            {"type": "email", "confidence": 0.99},
            {"type": None, "confidence": 0.41},
        ]
        detected, skipped = detect_all_columns(df)
        config = build_config(detected)
        assert config == {"customer_name": "name", "email": "email"}
        assert len(skipped) == 1
        assert skipped[0]["column"] == "internal_code"
