"""Tests: pipeline handles 10M rows without memory issues or timeouts.

These tests are skipped automatically if the test data files have not been
generated yet. Run generate_synthetic_base.py and inject_messiness.py first.
"""

import time
import tracemalloc
from pathlib import Path

import pandas as pd
import pytest

from anonymize import anonymize_dataframe

MESSY_PARQUET = Path(__file__).parent / "test_data" / "synthetic_messy_10m.parquet"
ANON_PARQUET = Path(__file__).parent / "test_data" / "synthetic_messy_10m_ANON.parquet"
CONFIG = {"name": "name", "email": "email", "country": "country"}

requires_test_data = pytest.mark.skipif(
    not MESSY_PARQUET.exists(),
    reason=f"Test data not found. Run generate_synthetic_base.py and inject_messiness.py first.",
)


@requires_test_data
def test_full_pipeline_completes_within_time_limit():
    """Full 10M-row anonymization should complete in under 5 minutes."""
    df = pd.read_parquet(MESSY_PARQUET)

    start = time.time()
    result = anonymize_dataframe(df, CONFIG)
    elapsed = time.time() - start

    assert len(result) == len(df), "Row count must be preserved"
    assert elapsed < 300, f"Pipeline took {elapsed:.1f}s, expected < 300s"
    print(f"\n  Elapsed: {elapsed:.1f}s for {len(df):,} rows")


@requires_test_data
def test_memory_usage_is_bounded():
    """Peak memory during anonymization should not exceed 2x the input size."""
    df = pd.read_parquet(MESSY_PARQUET)
    input_bytes = df.memory_usage(deep=True).sum()

    tracemalloc.start()
    anonymize_dataframe(df, CONFIG)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < input_bytes * 3, (
        f"Peak memory {peak / 1e9:.2f} GB exceeded 3x input size {input_bytes / 1e9:.2f} GB"
    )


@requires_test_data
def test_parquet_roundtrip_is_lossless():
    """Anonymized output written to parquet and read back should be identical."""
    df = pd.read_parquet(MESSY_PARQUET)
    result = anonymize_dataframe(df, CONFIG)

    result.to_parquet(ANON_PARQUET, compression="zstd", index=False)
    result_read = pd.read_parquet(ANON_PARQUET)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        result_read.reset_index(drop=True),
        check_like=True,
    )
