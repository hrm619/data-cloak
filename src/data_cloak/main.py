import argparse
import sys
from pathlib import Path

from data_cloak.config import load_config
from data_cloak.data_io import read_csv, anon_path, write_csv
from data_cloak.anonymize import anonymize_dataframe


def _run_with_config(input_path: Path, config_path: Path) -> None:
    """v1.0 path: load explicit config, anonymize, write output."""
    config = load_config(config_path)
    df = read_csv(input_path)
    result = anonymize_dataframe(df, config, filename=input_path.name)
    output = anon_path(input_path)
    write_csv(result, output)
    print(f"Wrote anonymized file to {output}")


def _run_with_detection(input_path: Path, model: str) -> None:
    """v1.1 path: detect columns, confirm, anonymize, save config + output."""
    from data_cloak.cli_ui import (
        check_ollama,
        display_proposed_config,
        prompt_confirmation,
        save_config_yaml,
        load_config_yaml,
        edit_config_in_editor,
        print_report,
    )
    from data_cloak.column_detector import detect_all_columns, build_config

    if not check_ollama():
        print("Ollama not found. Start Ollama or use --config to skip detection.")
        sys.exit(1)

    df = read_csv(input_path)

    def _progress(col, status):
        if status == "start":
            print(f"  Classifying {col}...", end=" ", flush=True)
        else:
            print("done")

    print()
    detected, skipped = detect_all_columns(df, model=model, on_progress=_progress)

    display_proposed_config(detected, skipped, input_path.name)

    if not detected:
        print("  No columns detected with sufficient confidence. Nothing to anonymize.")
        sys.exit(0)

    choice = prompt_confirmation()

    if choice == "quit":
        print("  Aborted. No output written.")
        sys.exit(0)

    config_yaml_path = input_path.parent / "datacloak_config.yaml"
    save_config_yaml(detected, skipped, input_path.name, config_yaml_path)

    if choice == "edit":
        config = edit_config_in_editor(config_yaml_path)
    else:
        config = build_config(detected)

    result = anonymize_dataframe(df, config, filename=input_path.name)
    output = anon_path(input_path)
    write_csv(result, output)
    print(f"  Wrote anonymized file to {output}")
    print(f"  Saved config to {config_yaml_path}")
    print_report(config, len(df), skipped)


def main():
    parser = argparse.ArgumentParser(description="Anonymize PII columns in a CSV file.")
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument(
        "--config", default=None,
        help="Path to a TOML or YAML config file. If omitted, auto-detection runs via Ollama.",
    )
    parser.add_argument(
        "--model", default="llama3.1:8b",
        help="Ollama model for column detection (default: llama3.1:8b).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {args.input}")

    if args.config is not None:
        _run_with_config(input_path, Path(args.config))
    else:
        _run_with_detection(input_path, args.model)


if __name__ == "__main__":
    main()
