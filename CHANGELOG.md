# Changelog

## v1.1.0 — 2026-03-14

### Added
- Automatic column detection via local LLM (Ollama). When no `--config` is provided, DataCloak samples the CSV, classifies each column, and proposes a config for user confirmation.
- New module `column_detector.py`: `sample_column`, `classify_column`, `detect_all_columns`, `build_config`.
- New module `cli_ui.py`: Ollama health check, proposed config display, [A]ccept/[E]dit/[Q]uit confirmation flow, YAML config save/load, post-run report.
- `--model` flag to override the default Ollama model (`llama3.1:8b`).
- YAML config support (`.yaml`/`.yml`) in `load_config`, alongside existing TOML.
- Generated `datacloak_config.yaml` saved alongside output, reusable with `--config`.
- 84 new tests across `test_column_detector.py` and `test_cli_integration.py` (159 total).

### Changed
- `--config` default changed from `config.toml` to `None`. Omitting it triggers auto-detection.
- `anonymize_column` now gracefully handles type mismatches (e.g., a string in an `amount` column) — values that fail conversion pass through unchanged instead of crashing.
- `config.py` refactored into `_load_toml`, `_load_yaml`, and `_extract_columns` atomic functions.
- `main.py` split into `_run_with_config` (v1.0 path) and `_run_with_detection` (v1.1 path).
- Version bumped to 1.1.0.
- Added `pyyaml>=6.0` to dependencies.

### Unchanged
- `--config` flag bypasses detection entirely — v1.0 behavior is fully preserved.
- All anonymization logic (`anonymize.py`) unchanged apart from the graceful error handling wrapper.
- All 75 existing tests pass without modification.

## v1.0.0

Initial release. Deterministic anonymization for names, emails, countries, dates, amounts, and transaction descriptions via explicit TOML config.
