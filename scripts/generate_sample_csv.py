"""Generate a small sample CSV with realistic PII columns for testing data-cloak."""

import argparse
import csv
import sys

from faker import Faker


def generate_rows(fake: Faker, n: int) -> list[dict]:
    """Generate n rows of sample PII data."""
    return [
        {
            "name": fake.name(),
            "email": fake.email(),
            "country": fake.country(),
            "created_at": fake.date_between("-2y", "today").strftime("%m/%d/%Y"),
            "revenue": f"{fake.pyfloat(min_value=10, max_value=5000, right_digits=2):.2f}",
            "merchant": fake.company(),
            "account_id": f"ACC-{fake.random_int(min=10000, max=99999)}",
        }
        for _ in range(n)
    ]


def write_csv(rows: list[dict], path: str) -> None:
    """Write rows to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate a sample CSV for data-cloak testing.")
    parser.add_argument("-n", "--rows", type=int, default=50, help="Number of rows (default: 50)")
    parser.add_argument("-o", "--output", default="test_sample.csv", help="Output path (default: test_sample.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Faker seed (default: 42)")
    args = parser.parse_args()

    fake = Faker()
    Faker.seed(args.seed)

    rows = generate_rows(fake, args.rows)
    write_csv(rows, args.output)
    print(f"Wrote {args.rows} rows to {args.output}")


if __name__ == "__main__":
    main()
