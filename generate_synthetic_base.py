#!/usr/bin/env python3
"""Generate a 10M-row synthetic dataset using Faker with multi-locale support.

Output: test_data/synthetic_base_10m.parquet (ZSTD compressed)
"""

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker

TOTAL_ROWS = 10_000_000
CHUNK_SIZE = 100_000
OUTPUT_DIR = Path(__file__).parent / "test_data"
OUTPUT_PATH = OUTPUT_DIR / "synthetic_base_10m.parquet"

# (locale, ISO 3166-1 alpha-2, weight) — roughly US 40%, EMEA 35%, APAC 25%
LOCALE_CONFIG = [
    ("en_US", "US", 0.40),
    ("de_DE", "DE", 0.04),
    ("fr_FR", "FR", 0.04),
    ("it_IT", "IT", 0.03),
    ("es_ES", "ES", 0.03),
    ("pt_BR", "BR", 0.03),
    ("nl_NL", "NL", 0.02),
    ("sv_SE", "SE", 0.01),
    ("no_NO", "NO", 0.01),
    ("da_DK", "DK", 0.01),
    ("fi_FI", "FI", 0.01),
    ("el_GR", "GR", 0.01),
    ("tr_TR", "TR", 0.02),
    ("pl_PL", "PL", 0.02),
    ("uk_UA", "UA", 0.01),
    ("cs_CZ", "CZ", 0.01),
    ("hu_HU", "HU", 0.01),
    ("ro_RO", "RO", 0.01),
    ("ja_JP", "JP", 0.05),
    ("zh_CN", "CN", 0.05),
    ("ko_KR", "KR", 0.03),
    ("th_TH", "TH", 0.02),
    ("hi_IN", "IN", 0.04),
    ("vi_VN", "VN", 0.01),
    ("id_ID", "ID", 0.02),
    ("ar_SA", "SA", 0.02),
    ("ru_RU", "RU", 0.03),
    ("sk_SK", "SK", 0.005),
    ("bg_BG", "BG", 0.005),
    ("hr_HR", "HR", 0.005),
    ("lt_LT", "LT", 0.005),
]


def build_fakers() -> tuple[list[Faker], list[float], list[str]]:
    """Instantiate one Faker per locale, skipping any that are unavailable."""
    fakers, weights, countries = [], [], []
    for locale, country, weight in LOCALE_CONFIG:
        try:
            fakers.append(Faker(locale))
            weights.append(weight)
            countries.append(country)
        except Exception as exc:
            print(f"  Skipping locale {locale}: {exc}")
    return fakers, weights, countries


def generate_chunk(
    fakers: list[Faker],
    weights: list[float],
    countries: list[str],
    size: int,
) -> dict[str, list[str]]:
    """Generate one chunk of synthetic rows."""
    indices = random.choices(range(len(fakers)), weights=weights, k=size)
    names, emails, country_list = [], [], []
    for idx in indices:
        faker = fakers[idx]
        names.append(faker.name())
        emails.append(faker.email())
        country_list.append(countries[idx])
    return {"name": names, "email": emails, "country": country_list}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fakers, weights, countries = build_fakers()
    print(f"Loaded {len(fakers)} locales.")

    schema = pa.schema([
        pa.field("name", pa.string()),
        pa.field("email", pa.string()),
        pa.field("country", pa.string()),
    ])

    rows_written = 0
    with pq.ParquetWriter(OUTPUT_PATH, schema, compression="zstd") as writer:
        while rows_written < TOTAL_ROWS:
            chunk_size = min(CHUNK_SIZE, TOTAL_ROWS - rows_written)
            chunk = generate_chunk(fakers, weights, countries, chunk_size)
            writer.write_table(pa.table(chunk, schema=schema))
            rows_written += chunk_size
            if rows_written % 1_000_000 == 0:
                print(f"  {rows_written:,} / {TOTAL_ROWS:,} rows written")

    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
