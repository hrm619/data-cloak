"""Tests for the v1.1 CLI integration: detection flow, config path, and UX."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import yaml

from data_cloak.cli_ui import (
    check_ollama,
    display_proposed_config,
    save_config_yaml,
    load_config_yaml,
    print_report,
)
from data_cloak.config import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_detected():
    return [
        {"column": "customer_name", "type": "name", "confidence": 0.97},
        {"column": "email", "type": "email", "confidence": 0.99},
    ]


@pytest.fixture
def sample_skipped():
    return [
        {"column": "internal_code", "type": None, "confidence": 0.41},
    ]


@pytest.fixture
def csv_path(tmp_path):
    df = pd.DataFrame({
        "customer_name": ["Alice", "Bob"],
        "email": ["a@b.com", "c@d.com"],
        "country": ["US", "DE"],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# check_ollama
# ---------------------------------------------------------------------------

class TestCheckOllama:

    @patch("data_cloak.cli_ui.urlopen")
    def test_returns_true_when_reachable(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        assert check_ollama() is True

    @patch("data_cloak.cli_ui.urlopen")
    def test_returns_false_on_connection_error(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("connection refused")
        assert check_ollama() is False

    @patch("data_cloak.cli_ui.urlopen")
    def test_returns_false_on_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        assert check_ollama() is False


# ---------------------------------------------------------------------------
# display_proposed_config
# ---------------------------------------------------------------------------

class TestDisplayProposedConfig:

    def test_prints_detected_columns(self, capsys, sample_detected, sample_skipped):
        display_proposed_config(sample_detected, sample_skipped, "test.csv")
        output = capsys.readouterr().out
        assert "customer_name" in output
        assert "email" in output
        assert "name" in output
        assert "0.97" in output
        assert "WILL ANONYMIZE (2 columns)" in output

    def test_prints_skipped_columns(self, capsys, sample_detected, sample_skipped):
        display_proposed_config(sample_detected, sample_skipped, "test.csv")
        output = capsys.readouterr().out
        assert "internal_code" in output
        assert "SKIPPED" in output
        assert "0.41" in output

    def test_prints_filename(self, capsys, sample_detected, sample_skipped):
        display_proposed_config(sample_detected, sample_skipped, "transactions.csv")
        output = capsys.readouterr().out
        assert "transactions.csv" in output

    def test_no_detected_no_will_anonymize(self, capsys, sample_skipped):
        display_proposed_config([], sample_skipped, "test.csv")
        output = capsys.readouterr().out
        assert "WILL ANONYMIZE" not in output

    def test_no_skipped_no_skipped_section(self, capsys, sample_detected):
        display_proposed_config(sample_detected, [], "test.csv")
        output = capsys.readouterr().out
        assert "SKIPPED" not in output


# ---------------------------------------------------------------------------
# save_config_yaml / load_config_yaml round-trip
# ---------------------------------------------------------------------------

class TestSaveAndLoadConfigYaml:

    def test_save_creates_file(self, tmp_path, sample_detected, sample_skipped):
        path = tmp_path / "datacloak_config.yaml"
        save_config_yaml(sample_detected, sample_skipped, "test.csv", path)
        assert path.exists()

    def test_saved_file_is_valid_yaml(self, tmp_path, sample_detected, sample_skipped):
        path = tmp_path / "datacloak_config.yaml"
        save_config_yaml(sample_detected, sample_skipped, "test.csv", path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "columns" in data

    def test_round_trip_preserves_types(self, tmp_path, sample_detected, sample_skipped):
        path = tmp_path / "datacloak_config.yaml"
        save_config_yaml(sample_detected, sample_skipped, "test.csv", path)
        config = load_config_yaml(path)
        assert config == {"customer_name": "name", "email": "email"}

    def test_skipped_section_saved(self, tmp_path, sample_detected, sample_skipped):
        path = tmp_path / "datacloak_config.yaml"
        save_config_yaml(sample_detected, sample_skipped, "test.csv", path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "skipped" in data
        assert "internal_code" in data["skipped"]

    def test_header_comment_includes_filename(self, tmp_path, sample_detected, sample_skipped):
        path = tmp_path / "datacloak_config.yaml"
        save_config_yaml(sample_detected, sample_skipped, "test.csv", path)
        text = path.read_text()
        assert "# Source: test.csv" in text
        assert "auto-generated by v1.1" in text

    def test_no_skipped_omits_section(self, tmp_path, sample_detected):
        path = tmp_path / "datacloak_config.yaml"
        save_config_yaml(sample_detected, [], "test.csv", path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "skipped" not in data


# ---------------------------------------------------------------------------
# load_config supports YAML (backward compat with TOML)
# ---------------------------------------------------------------------------

class TestLoadConfigYamlCompat:

    def test_load_config_reads_yaml(self, tmp_path, sample_detected, sample_skipped):
        path = tmp_path / "config.yaml"
        save_config_yaml(sample_detected, sample_skipped, "test.csv", path)
        config = load_config(path)
        assert config == {"customer_name": "name", "email": "email"}

    def test_load_config_reads_yml_extension(self, tmp_path, sample_detected, sample_skipped):
        path = tmp_path / "config.yml"
        save_config_yaml(sample_detected, sample_skipped, "test.csv", path)
        config = load_config(path)
        assert config == {"customer_name": "name", "email": "email"}

    def test_load_config_still_reads_toml(self):
        config = load_config("config.toml")
        assert "Transaction Date" in config or "Amount" in config


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------

class TestPrintReport:

    def test_shows_column_count(self, capsys):
        config = {"email": "email", "name": "name"}
        print_report(config, 100, [])
        output = capsys.readouterr().out
        assert "Columns anonymized: 2" in output

    def test_shows_row_count(self, capsys):
        print_report({"email": "email"}, 500, [])
        output = capsys.readouterr().out
        assert "Rows processed:     500" in output

    def test_shows_skipped_names(self, capsys):
        skipped = [{"column": "notes", "type": None, "confidence": 0.3}]
        print_report({"email": "email"}, 10, skipped)
        output = capsys.readouterr().out
        assert "notes" in output


# ---------------------------------------------------------------------------
# prompt_confirmation
# ---------------------------------------------------------------------------

class TestPromptConfirmation:

    @patch("builtins.input", return_value="a")
    def test_accept(self, _):
        from data_cloak.cli_ui import prompt_confirmation
        assert prompt_confirmation() == "accept"

    @patch("builtins.input", return_value="e")
    def test_edit(self, _):
        from data_cloak.cli_ui import prompt_confirmation
        assert prompt_confirmation() == "edit"

    @patch("builtins.input", return_value="q")
    def test_quit(self, _):
        from data_cloak.cli_ui import prompt_confirmation
        assert prompt_confirmation() == "quit"

    @patch("builtins.input", side_effect=["x", "A"])
    def test_rejects_invalid_then_accepts(self, _):
        from data_cloak.cli_ui import prompt_confirmation
        assert prompt_confirmation() == "accept"


# ---------------------------------------------------------------------------
# CLI entry point: --config path (v1.0 backward compat)
# ---------------------------------------------------------------------------

class TestCLIConfigPath:

    def test_config_flag_runs_v1_path(self, csv_path):
        """--config skips detection entirely and produces output."""
        # Create a minimal TOML config for the test CSV
        config_path = csv_path.parent / "test_config.toml"
        config_path.write_text('[columns]\ncustomer_name = "name"\nemail = "email"\n')
        result = subprocess.run(
            ["uv", "run", "data-cloak", str(csv_path), "--config", str(config_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Wrote anonymized file" in result.stdout
        anon_file = csv_path.parent / "test_data_ANON.csv"
        assert anon_file.exists()

    def test_config_flag_with_yaml(self, csv_path, sample_detected, sample_skipped):
        """--config also works with YAML files generated by v1.1."""
        yaml_path = csv_path.parent / "config.yaml"
        save_config_yaml(sample_detected, sample_skipped, "test_data.csv", yaml_path)
        result = subprocess.run(
            ["uv", "run", "data-cloak", str(csv_path), "--config", str(yaml_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Wrote anonymized file" in result.stdout

    def test_missing_input_file_errors(self):
        result = subprocess.run(
            ["uv", "run", "data-cloak", "/nonexistent.csv", "--config", "config.toml"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr


# ---------------------------------------------------------------------------
# CLI entry point: detection path (no --config)
# ---------------------------------------------------------------------------

class TestCLIDetectionPath:

    @patch("data_cloak.cli_ui.check_ollama", return_value=False)
    def test_no_config_without_ollama_exits_cleanly(self, _ollama, csv_path, capsys):
        """When Ollama isn't running and no --config, exit with clear error."""
        with pytest.raises(SystemExit) as exc_info:
            from data_cloak.main import _run_with_detection
            _run_with_detection(csv_path, "llama3.1:8b")
        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "Ollama not found" in output


# ---------------------------------------------------------------------------
# End-to-end: detection → accept → anonymize (mocked Ollama)
# ---------------------------------------------------------------------------

class TestEndToEndDetectionFlow:

    @patch("data_cloak.column_detector.classify_column")
    @patch("data_cloak.cli_ui.check_ollama", return_value=True)
    @patch("builtins.input", return_value="a")
    def test_accept_flow_produces_output(self, _input, _ollama, mock_classify, csv_path):
        mock_classify.side_effect = [
            {"type": "name", "confidence": 0.97},
            {"type": "email", "confidence": 0.99},
            {"type": "country", "confidence": 0.91},
        ]

        from data_cloak.main import _run_with_detection
        _run_with_detection(csv_path, "llama3.1:8b")

        anon_file = csv_path.parent / "test_data_ANON.csv"
        assert anon_file.exists()
        config_file = csv_path.parent / "datacloak_config.yaml"
        assert config_file.exists()

        # Verify anonymized output has same row count
        result_df = pd.read_csv(anon_file)
        assert len(result_df) == 2

    @patch("data_cloak.column_detector.classify_column")
    @patch("data_cloak.cli_ui.check_ollama", return_value=True)
    @patch("builtins.input", return_value="q")
    def test_quit_flow_no_output(self, _input, _ollama, mock_classify, csv_path):
        mock_classify.side_effect = [
            {"type": "name", "confidence": 0.97},
            {"type": "email", "confidence": 0.99},
            {"type": "country", "confidence": 0.91},
        ]

        with pytest.raises(SystemExit) as exc_info:
            from data_cloak.main import _run_with_detection
            _run_with_detection(csv_path, "llama3.1:8b")

        assert exc_info.value.code == 0
        anon_file = csv_path.parent / "test_data_ANON.csv"
        assert not anon_file.exists()

    @patch("data_cloak.column_detector.classify_column")
    @patch("data_cloak.cli_ui.check_ollama", return_value=True)
    @patch("builtins.input", return_value="a")
    def test_low_confidence_columns_skipped(self, _input, _ollama, mock_classify, csv_path):
        mock_classify.side_effect = [
            {"type": "name", "confidence": 0.97},
            {"type": "email", "confidence": 0.99},
            {"type": None, "confidence": 0.30},  # country skipped
        ]

        from data_cloak.main import _run_with_detection
        _run_with_detection(csv_path, "llama3.1:8b")

        config_file = csv_path.parent / "datacloak_config.yaml"
        config = load_config_yaml(config_file)
        assert "country" not in config
        assert "customer_name" in config
        assert "email" in config

    @patch("data_cloak.column_detector.classify_column")
    @patch("data_cloak.cli_ui.check_ollama", return_value=True)
    def test_no_detected_columns_exits(self, _ollama, mock_classify, csv_path):
        mock_classify.return_value = {"type": None, "confidence": 0.1}

        with pytest.raises(SystemExit) as exc_info:
            from data_cloak.main import _run_with_detection
            _run_with_detection(csv_path, "llama3.1:8b")

        assert exc_info.value.code == 0

    @patch("data_cloak.cli_ui.check_ollama", return_value=False)
    def test_ollama_not_running_exits(self, _ollama, csv_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            from data_cloak.main import _run_with_detection
            _run_with_detection(csv_path, "llama3.1:8b")

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "Ollama not found" in output


# ---------------------------------------------------------------------------
# Generated config is reusable as --config input
# ---------------------------------------------------------------------------

class TestGeneratedConfigReuse:

    def test_saved_yaml_works_as_config_input(self, csv_path, sample_detected, sample_skipped):
        yaml_path = csv_path.parent / "datacloak_config.yaml"
        save_config_yaml(sample_detected, sample_skipped, "test_data.csv", yaml_path)

        config = load_config(yaml_path)
        df = pd.read_csv(csv_path)

        from data_cloak.anonymize import anonymize_dataframe
        result = anonymize_dataframe(df, config, filename="test_data.csv")
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)
