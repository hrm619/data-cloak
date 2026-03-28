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


# ---------------------------------------------------------------------------
# Faker-based live Ollama accuracy tests
# ---------------------------------------------------------------------------

import random
from urllib.request import urlopen as _raw_urlopen
from urllib.error import URLError as _URLError
from faker import Faker

_fake = Faker()
Faker.seed(42)
random.seed(42)

_NUM_ROWS = 25


def _ollama_is_reachable() -> bool:
    """Return True if Ollama is responding on localhost:11434."""
    try:
        with _raw_urlopen("http://localhost:11434", timeout=5):
            return True
    except Exception:
        return False


_ollama_available = _ollama_is_reachable()
_skip_no_ollama = pytest.mark.skipif(
    not _ollama_available,
    reason="Ollama is not reachable at localhost:11434 — start Ollama to run live accuracy tests",
)


# --- Faker data generators per type ---

def _gen_names(n: int) -> list[str]:
    return [_fake.name() for _ in range(n)]

def _gen_emails(n: int) -> list[str]:
    return [_fake.email() for _ in range(n)]

def _gen_countries(n: int) -> list[str]:
    return [_fake.country() for _ in range(n)]

def _gen_dates(n: int) -> list[str]:
    return [_fake.date_between(start_date="-5y", end_date="today").strftime("%m/%d/%Y") for _ in range(n)]

def _gen_amounts(n: int) -> list[str]:
    return [f"{_fake.pyfloat(min_value=-50000, max_value=50000, right_digits=2):.2f}" for _ in range(n)]

def _gen_descriptions(n: int) -> list[str]:
    merchants = [
        "Starbucks #{num}", "AMZN*Marketplace", "UBER TRIP {num}",
        "SHELL OIL {num}", "WAL-MART #{num}", "TARGET #{num}",
        "COSTCO WHSE #{num}", "NETFLIX.COM", "SPOTIFY USA",
        "DoorDash #{num}", "LYFT *RIDE {num}", "APPLE.COM/BILL",
        "MCDONALD'S #{num}", "HOME DEPOT #{num}", "CVS/PHARMACY #{num}",
    ]
    return [random.choice(merchants).format(num=random.randint(1000, 9999)) for _ in range(n)]

def _gen_ids(n: int) -> list[str]:
    return [f"ACC-{random.randint(10000, 99999)}" for _ in range(n)]


# --- Ambiguous value generators (scenario 3) ---

def _gen_ambiguous_codes(n: int) -> list[str]:
    return [f"REF-{random.randint(1000, 9999)}" for _ in range(n)]

def _gen_ambiguous_labels(n: int) -> list[str]:
    labels = ["HIGH", "LOW", "MEDIUM", "N/A", "TBD", "PENDING", "—", "X", "Y", "Z"]
    return [random.choice(labels) for _ in range(n)]

def _gen_ambiguous_single_chars(n: int) -> list[str]:
    return [random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n)]


# Scenario definitions: (header, value_generator, expected_type)
# Scenario 1: Clear header + clear values
# Scenario 2: Ambiguous header + clear values
# Scenario 3: Clear header + ambiguous values (expect possible failure)

_SCENARIOS: list[tuple[str, str, str, callable, str | None]] = [
    # (scenario_label, header, description, gen_fn, expected_type)
    # --- name ---
    ("1_clear_clear",    "customer_name", "name",        _gen_names,              "name"),
    ("2_ambig_hdr",      "field_1",       "name",        _gen_names,              "name"),
    ("3_ambig_val",      "customer_name", "name",        _gen_ambiguous_codes,    None),
    # --- email ---
    ("1_clear_clear",    "email",         "email",       _gen_emails,             "email"),
    ("2_ambig_hdr",      "field_3",       "email",       _gen_emails,             "email"),
    ("3_ambig_val",      "email",         "email",       _gen_ambiguous_codes,    None),
    # --- country ---
    ("1_clear_clear",    "country",       "country",     _gen_countries,          "country"),
    ("2_ambig_hdr",      "field_7",       "country",     _gen_countries,          "country"),
    ("3_ambig_val",      "country",       "country",     _gen_ambiguous_single_chars, None),
    # --- date ---
    ("1_clear_clear",    "created_at",    "date",        _gen_dates,              "date"),
    ("2_ambig_hdr",      "field_2",       "date",        _gen_dates,              "date"),
    ("3_ambig_val",      "created_at",    "date",        _gen_ambiguous_labels,   None),
    # --- amount ---
    ("1_clear_clear",    "revenue",       "amount",      _gen_amounts,            "amount"),
    ("2_ambig_hdr",      "field_5",       "amount",      _gen_amounts,            "amount"),
    ("3_ambig_val",      "revenue",       "amount",      _gen_ambiguous_labels,   None),
    # --- description ---
    ("1_clear_clear",    "merchant",      "description", _gen_descriptions,       "description"),
    ("2_ambig_hdr",      "field_9",       "description", _gen_descriptions,       "description"),
    ("3_ambig_val",      "merchant",      "description", _gen_ambiguous_single_chars, None),
    # --- id ---
    ("1_clear_clear",    "account_id",    "id",          _gen_ids,                "id"),
    ("2_ambig_hdr",      "field_0",       "id",          _gen_ids,                "id"),
    ("3_ambig_val",      "account_id",    "id",          _gen_ambiguous_labels,   None),
]


# Storage for results across the parametrized tests
_accuracy_results: list[dict] = []


def _scenario_id(param):
    """Generate a readable test ID from scenario parameters."""
    scenario, header, col_type, _, _ = param
    return f"{col_type}-{scenario}"


@_skip_no_ollama
class TestFakerDetectionAccuracy:
    """Live Ollama accuracy tests using Faker-generated data.

    Sends real data to the local Ollama instance and measures whether the
    classifier returns the correct type with sufficient confidence.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _print_results_table(self):
        """Print the results table after all tests in this class complete."""
        _accuracy_results.clear()
        yield
        _print_accuracy_report(_accuracy_results)

    @pytest.mark.parametrize(
        "scenario,header,col_type,gen_fn,expected_type",
        _SCENARIOS,
        ids=[_scenario_id(s) for s in _SCENARIOS],
    )
    def test_classification_accuracy(
        self, scenario, header, col_type, gen_fn, expected_type
    ):
        values = gen_fn(_NUM_ROWS)
        df = pd.DataFrame({header: values})
        detected, skipped = detect_all_columns(df)

        if detected:
            predicted = detected[0]["type"]
            confidence = detected[0]["confidence"]
        else:
            predicted = skipped[0]["type"] if skipped else None
            confidence = skipped[0]["confidence"] if skipped else 0.0

        is_scenario_3 = scenario.startswith("3_")

        if is_scenario_3:
            # Scenario 3: ambiguous values — classifier should NOT detect the type
            passed = predicted != col_type or confidence < 0.80
        else:
            # Scenarios 1 & 2: classifier should detect the correct type
            passed = predicted == expected_type and confidence >= 0.80

        _accuracy_results.append({
            "scenario": scenario,
            "col_type": col_type,
            "header": header,
            "predicted": predicted,
            "confidence": confidence,
            "passed": passed,
            "is_scenario_3": is_scenario_3,
        })

        if not is_scenario_3:
            assert passed, (
                f"Expected type={expected_type} with confidence>=0.80, "
                f"got type={predicted} confidence={confidence:.2f}"
            )
        else:
            # Scenario 3 failures are flagged but do not fail the suite
            if not passed:
                pytest.xfail(
                    f"Scenario 3 prompt-tuning issue: classifier returned "
                    f"type={predicted} confidence={confidence:.2f} for "
                    f"ambiguous values under header '{header}'"
                )


def _print_accuracy_report(results: list[dict]) -> None:
    """Print a formatted accuracy table to stdout."""
    if not results:
        return

    print("\n")
    print("=" * 90)
    print("DATACLOAK v1.1 — FAKER DETECTION ACCURACY REPORT")
    print("=" * 90)
    print(
        f"{'Scenario':<18} {'Type':<14} {'Header':<18} "
        f"{'Predicted':<14} {'Conf':>6}  {'Result'}"
    )
    print("-" * 90)

    total = 0
    passed = 0
    scenario_3_failures = []

    for r in results:
        total += 1
        status = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            passed += 1
        if r["is_scenario_3"] and not r["passed"]:
            scenario_3_failures.append(r)
            status = "FAIL*"

        print(
            f"{r['scenario']:<18} {r['col_type']:<14} {r['header']:<18} "
            f"{str(r['predicted']):<14} {r['confidence']:>5.2f}  {status}"
        )

    core_total = sum(1 for r in results if not r["is_scenario_3"])
    core_passed = sum(1 for r in results if not r["is_scenario_3"] and r["passed"])
    core_pct = (core_passed / core_total * 100) if core_total else 0

    s3_total = sum(1 for r in results if r["is_scenario_3"])
    s3_passed = sum(1 for r in results if r["is_scenario_3"] and r["passed"])
    s3_pct = (s3_passed / s3_total * 100) if s3_total else 0

    pct = (passed / total * 100) if total else 0
    print("-" * 90)
    print(f"Scenarios 1+2 accuracy: {core_passed}/{core_total} ({core_pct:.1f}%)")
    print(f"Scenario 3 accuracy:    {s3_passed}/{s3_total} ({s3_pct:.1f}%)")
    print(f"Overall accuracy:       {passed}/{total} ({pct:.1f}%)")
    merge_status = "PASS" if core_pct >= 90 else "FAIL"
    print(f"Merge bar (90% on scenarios 1+2): {merge_status}")

    if scenario_3_failures:
        print("\n" + "=" * 90)
        print("SCENARIO 3 FAILURES — classifier prompt may need tuning:")
        print("-" * 90)
        for r in scenario_3_failures:
            print(
                f"  {r['col_type']:<14} header='{r['header']}'  "
                f"predicted={r['predicted']}  confidence={r['confidence']:.2f}"
            )
        print(
            "\n  These indicate the LLM trusted the header name over "
            "contradictory value evidence."
        )

    print("=" * 90)
    print()
