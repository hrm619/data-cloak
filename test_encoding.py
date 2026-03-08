"""Tests: correct handling of non-ASCII characters and encoding edge cases."""

import pandas as pd
import pytest

from anonymize import anonymize_dataframe, anonymize_value

CONFIG = {"name": "name", "email": "email", "country": "country"}


def test_diacritics_treated_as_distinct_from_ascii():
    """'José' and 'Jose' are different inputs and should produce different outputs."""
    result_accented = anonymize_value("José", "name")
    result_plain = anonymize_value("Jose", "name")
    assert result_accented != result_plain, (
        "Accented and non-accented versions should hash differently"
    )


def test_non_latin_scripts_do_not_crash():
    """Chinese, Arabic, and Cyrillic names should anonymize without raising."""
    df = pd.DataFrame([
        {"name": "李明",      "email": "li@example.cn",       "country": "CN"},
        {"name": "محمد",      "email": "mohammad@example.sa", "country": "SA"},
        {"name": "Владимир", "email": "vlad@example.ru",      "country": "RU"},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert result["name"].notna().all(), "Non-Latin names should produce non-null anonymized values"
    assert result["email"].notna().all(), "Emails paired with non-Latin names should anonymize"


def test_mixed_script_names_do_not_crash():
    """Names mixing Latin and non-Latin characters should anonymize without raising."""
    df = pd.DataFrame([
        {"name": "田中 Tanaka", "email": "tanaka@example.jp", "country": "JP"},
        {"name": "José María", "email": "jm@example.mx",      "country": "MX"},
    ])
    result = anonymize_dataframe(df, CONFIG)

    assert result["name"].notna().all(), "Mixed-script names should produce non-null anonymized values"


def test_emoji_in_name_handled_gracefully():
    """Names containing emoji should either anonymize cleanly or raise a clear error."""
    df = pd.DataFrame([
        {"name": "John😀", "email": "john@example.com", "country": "US"},
    ])
    try:
        result = anonymize_dataframe(df, CONFIG)
        # If it succeeds, the output should be a non-null string
        assert pd.notna(result.iloc[0]["name"]), "Emoji name should produce a non-null result"
    except (ValueError, UnicodeEncodeError) as exc:
        msg = str(exc).lower()
        assert any(kw in msg for kw in ("emoji", "unicode", "encode")), (
            f"Error message should mention encoding/unicode, got: {exc}"
        )


def test_encoding_df_no_crash(encoding_df, config):
    """The full encoding fixture (11 edge-case rows) should not crash."""
    # Separate emoji row from the rest for the graceful-or-raise check
    non_emoji = encoding_df[~encoding_df["name"].str.contains("😀", na=False)]
    result = anonymize_dataframe(non_emoji, config)
    assert len(result) == len(non_emoji)
    assert result["name"].notna().all()


@pytest.mark.parametrize("name", [
    "José", "François", "Müller", "Søren", "李明", "محمد", "Владимир",
])
def test_specific_non_ascii_names_anonymize(name):
    """Each known non-ASCII name should anonymize to a non-null value."""
    result = anonymize_value(name, "name")
    assert isinstance(result, str) and len(result) > 0
