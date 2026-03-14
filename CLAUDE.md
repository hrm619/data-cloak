# CLAUDE.md

# Core Design Principle
we want to write our code as atomic functions such that each function does one thing only - ignorant of everything outside its responsibility. Compose small atomic functions into layers. One input, one output, one job. If a function reads, transforms, AND writes - split it.

# Intro
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`data-cloak` is a Python library for deterministic data anonymization. It replaces real PII values (names, emails, countries, dates, amounts, descriptions) with consistent fake substitutes using MD5-based hashing — the same input always maps to the same fake output.

As of v1.1, it can automatically detect which columns contain PII using a local LLM via Ollama, proposing a config for user confirmation before anonymizing.

## Setup & Commands

This project uses `uv` for package management with Python 3.13.

```bash
# Install dependencies (runtime + dev)
uv sync --all-extras

# Run with auto-detection (requires Ollama)
uv run data-cloak input.csv

# Run with explicit config (no Ollama needed)
uv run data-cloak input.csv --config financial_config.toml

# Run with a different Ollama model
uv run data-cloak input.csv --model mistral

# Run tests
uv run pytest tests/

# Run a single test file or class
uv run pytest tests/test_financial.py
uv run pytest tests/test_financial.py::TestAnonymizeDate

# Generate 10M-row test data (takes several minutes)
uv run python scripts/generate_synthetic_base.py
uv run python scripts/inject_messiness.py

# Run volume/performance tests (requires generated test data)
uv run pytest tests/test_volume_performance.py -v
```

## Architecture

Source code is in `src/data_cloak/`, tests in `tests/`, data generation scripts in `scripts/`.

**`src/data_cloak/anonymize.py`** — core anonymization logic, layered as atomic functions:
- `_md5_index(value, list_length)` — hashes a string to a stable list index; drives all determinism
- `derive_offset(filename)` → deterministic int in [180, 730] for date shifting
- `derive_multiplier(filename)` → deterministic float in [1.5, 3.5] for amount scaling
- `anonymize_date(value, offset)` — shifts MM/DD/YYYY date forward by offset days
- `anonymize_amount(value, multiplier)` — scales amount, preserves sign, rounds to 2dp
- `anonymize_description(value, category)` — replaces with synthetic merchant name from `_DESCRIPTIONS` pool (10 categories)
- `anonymize_value(value, field_type, **kwargs)` — dispatcher for all field types: `"name"`, `"email"`, `"country"`, `"date"`, `"amount"`, `"description"`
- `_safe_anonymize_value(value, field_type, **kwargs)` — wraps `anonymize_value` with error handling; returns value unchanged on type mismatch
- `anonymize_column(column, field_type, **kwargs)` — applies `_safe_anonymize_value` over a pandas Series, passing NULLs through
- `anonymize_description_column(description_col, category_col)` — cross-column description anonymization using a paired category column
- `anonymize_dataframe(df, config, filename=None)` — orchestrator; derives offset/multiplier from filename, handles both simple string and dict config entries
- Lookup pools: `_NAMES`, `_EMAIL_USERS`, `_EMAIL_DOMAINS`, `_COUNTRIES`, `_DESCRIPTIONS`

**`src/data_cloak/column_detector.py`** — automatic column classification (v1.1):
- `sample_column(df, col, pct)` — random sample of non-null values (10–20%, max 50)
- `classify_column(header, sample, model)` — sends prompt to Ollama `/api/generate`, returns `{type, confidence}`. Validates against `_VALID_TYPES`. Retries once on malformed JSON, 30s timeout.
- `detect_all_columns(df, model, threshold, on_progress)` — iterates columns sequentially, splits into detected (>= 0.80 confidence) and skipped lists
- `build_config(detected)` — flattens detected list into `{column: type}` dict for the anonymizer

**`src/data_cloak/cli_ui.py`** — terminal UI for the detection flow:
- `check_ollama()` — pings localhost:11434 to verify Ollama is running
- `display_proposed_config(detected, skipped, filename)` — prints the proposed config table
- `prompt_confirmation()` — [A]ccept / [E]dit / [Q]uit prompt, returns choice string
- `save_config_yaml(detected, skipped, filename, output_path)` — writes YAML config with header comments
- `load_config_yaml(path)` — reads YAML config back into `{column: type}` dict
- `edit_config_in_editor(config_path)` — opens config in $EDITOR, returns parsed result
- `print_report(config, row_count, skipped)` — prints post-anonymization summary

**`src/data_cloak/data_io.py`** — CSV I/O:
- `read_csv(path)` → DataFrame
- `write_csv(df, path)` → None
- `anon_path(path)` → derives output path, e.g. `users.csv` → `users_ANON.csv`

**`src/data_cloak/config.py`** — config loading (TOML and YAML):
- `load_config(path)` → detects format by extension (.yaml/.yml → YAML, else TOML), reads file, returns `dict[str, str | dict]`
- `_load_toml(path)`, `_load_yaml(path)` — format-specific loaders
- `_extract_columns(data, path)` — normalizes the columns table, stripping confidence metadata from YAML configs

**`config.toml`** — user-edited mapping of CSV column names to field types. Values can be simple strings (`"name"`) or inline tables for parameterized types (`{type = "description", category_column = "Category"}`).

**`src/data_cloak/main.py`** — CLI entry point with two paths:
- `_run_with_config(input_path, config_path)` — v1.0 path: load config → anonymize → write
- `_run_with_detection(input_path, model)` — v1.1 path: check Ollama → detect → confirm → anonymize → save config + output
- CLI flags: `input` (positional), `--config` (bypasses detection), `--model` (default: llama3.1:8b)

**`scripts/`** — test data generation (not part of the package):
- `generate_synthetic_base.py` — 10M-row synthetic dataset via Faker (31 weighted locales), output as ZSTD-compressed parquet
- `inject_messiness.py` — 17 atomic corruption functions (NULLs, duplicates, encoding edge cases, etc.)

**`tests/`** (9 files, 159 tests):
- `test_determinism.py`, `test_distribution.py`, `test_collisions.py`, `test_encoding.py`, `test_nulls.py` — core PII tests
- `test_financial.py` — date shifting, amount scaling, description anonymization, end-to-end financial pipeline
- `test_column_detector.py` — sampling, classification, detection, config building, signal scenarios (3 per type)
- `test_cli_integration.py` — Ollama check, UI display, YAML round-trip, confirmation flow, end-to-end detection, backward compatibility
- `test_volume_performance.py` — 10M-row pipeline time, memory, parquet roundtrip (auto-skipped if test data not generated)
