"""Shared pytest fixtures for data-cloak tests."""

from pathlib import Path

import pandas as pd
import pytest

CONFIG = {"name": "name", "email": "email", "country": "country"}


@pytest.fixture
def config():
    return CONFIG


@pytest.fixture
def small_df():
    """100 clean rows for determinism and basic unit tests."""
    return pd.DataFrame([
        {"name": f"Person {i}", "email": f"person{i}@example.com", "country": "US"}
        for i in range(100)
    ])


@pytest.fixture
def null_df():
    """DataFrame with intentional NULLs in each PII column."""
    return pd.DataFrame([
        {"name": None,        "email": "test@example.com",  "country": "US"},
        {"name": "John",      "email": None,                 "country": "US"},
        {"name": "Jane",      "email": "jane@example.com",  "country": None},
        {"name": None,        "email": None,                 "country": None},
        {"name": "Alice",     "email": "alice@example.com", "country": "DE"},
    ])


@pytest.fixture
def collision_df():
    """DataFrame constructed to exercise collision scenarios."""
    return pd.DataFrame([
        # Exact duplicates
        {"name": "John Smith", "email": "john@example.com", "country": "US"},
        {"name": "John Smith", "email": "john@example.com", "country": "US"},
        # Same name, different emails
        {"name": "John Smith", "email": "john1@example.com", "country": "US"},
        {"name": "John Smith", "email": "john2@example.com", "country": "US"},
        # Same email, different names
        {"name": "Jane Smith",  "email": "contact@example.com", "country": "US"},
        {"name": "James Smith", "email": "contact@example.com", "country": "US"},
    ])


@pytest.fixture
def encoding_df():
    """DataFrame with non-ASCII, mixed-script, and emoji edge cases."""
    return pd.DataFrame([
        {"name": "José",          "email": "jose@example.com",      "country": "ES"},
        {"name": "Jose",          "email": "jose2@example.com",     "country": "ES"},
        {"name": "François",      "email": "francois@example.fr",   "country": "FR"},
        {"name": "Müller",        "email": "muller@example.de",     "country": "DE"},
        {"name": "Søren",         "email": "soren@example.dk",      "country": "DK"},
        {"name": "李明",           "email": "li@example.cn",         "country": "CN"},
        {"name": "محمد",           "email": "mohammad@example.sa",   "country": "SA"},
        {"name": "Владимир",      "email": "vladimir@example.ru",   "country": "RU"},
        {"name": "田中 Tanaka",   "email": "tanaka@example.jp",     "country": "JP"},
        {"name": "José María",    "email": "jm@example.mx",         "country": "MX"},
        {"name": "John😀",        "email": "john@example.com",      "country": "US"},
    ])
