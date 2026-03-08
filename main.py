import argparse
from pathlib import Path

from config import load_config
from data_io import read_csv, anon_path, write_csv
from anonymize import anonymize_dataframe


def main():
    parser = argparse.ArgumentParser(description="Anonymize PII columns in a CSV file.")
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument("--config", default="config.toml", help="Path to the TOML config file (default: config.toml).")
    args = parser.parse_args()

    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")

    config = load_config(args.config)
    df = read_csv(args.input)
    result = anonymize_dataframe(df, config, filename=Path(args.input).name)
    output = anon_path(args.input)
    write_csv(result, output)
    print(f"Wrote anonymized file to {output}")


if __name__ == "__main__":
    main()
