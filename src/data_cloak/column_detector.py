import json
from urllib.request import urlopen, Request
from urllib.error import URLError

import pandas as pd


_OLLAMA_URL = "http://localhost:11434/api/generate"
_TIMEOUT = 30
_MAX_SAMPLE = 50
_VALID_TYPES = {"name", "email", "country", "date", "amount", "description", "id"}

_CLASSIFIER_PROMPT = """You are a data classification assistant. Classify the following CSV column into exactly one type.

Column header: {header}
Sample values (up to 50): {values}

Supported types and what they mean:
- name: human person names, typically first + last name (e.g. "Sarah Johnson", "J. Smith", "María García"). Names are personal identifiers for individual people.
- email: email addresses containing @ (e.g. "alice@example.com")
- country: country names, nationalities, or ISO country codes (e.g. "United States", "France", "Germany", "BR", "JP"). Note: country names are geopolitical entities, not person names — if most values are recognized countries, classify as country.
- date: calendar dates in any format (e.g. "03/15/2024", "2023-12-01")
- amount: numeric monetary values, possibly negative (e.g. "1204.50", "-89.00")
- description: transaction descriptions, merchant names, or business names — these often contain store names, numbers, symbols, or codes (e.g. "Starbucks #4821", "AMZN*Marketplace", "UBER TRIP 1234", "WAL-MART #5012")
- id: identifiers, codes, or reference numbers (e.g. "ACC-00192", "TXN-78432", "123-45-6789")

CRITICAL RULES:
1. Classify based on the ACTUAL VALUES, not the column header. The header may be generic (e.g. "field_1") or misleading.
2. If the values clearly match a type, classify accordingly regardless of the header.
3. If the values DO NOT match what the header suggests, trust the values. For example, if the header says "email" but the values are codes like "REF-1234", respond with null.
4. Lower your confidence when the values are ambiguous or do not clearly fit a single type.

Respond with JSON only. No explanation. Example:
{{"type": "email", "confidence": 0.95}}

If the values do not clearly match any supported type, respond:
{{"type": null, "confidence": 0.0}}"""


def sample_column(df: pd.DataFrame, col: str, pct: float = 0.1) -> list[str]:
    """Return a random sample of non-null values from a column.

    Args:
        df: Source DataFrame.
        col: Column name to sample.
        pct: Fraction of rows to sample (0.1–0.2). Clamped to [0.1, 0.2].

    Returns:
        List of stringified non-null values, capped at 50.
    """
    pct = max(0.1, min(0.2, pct))
    series = df[col].dropna()
    n = min(len(series), max(1, int(len(series) * pct)))
    n = min(n, _MAX_SAMPLE)
    return [str(v) for v in series.sample(n=n, random_state=42)]


def classify_column(
    header: str, sample: list[str], model: str = "llama3.1:8b"
) -> dict:
    """Send a classification prompt to Ollama and return the result.

    Args:
        header: Column header name.
        sample: List of sample value strings.
        model: Ollama model name.

    Returns:
        Dict with "type" (str or None) and "confidence" (float).
    """
    prompt = _CLASSIFIER_PROMPT.format(
        header=header, values=", ".join(sample)
    )
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    request = Request(
        _OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(2):
        try:
            with urlopen(request, timeout=_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            error_body = ""
            if hasattr(e, "read"):
                error_body = e.read().decode("utf-8", errors="replace")
            if "model" in error_body.lower() and "not found" in error_body.lower():
                raise RuntimeError(
                    f"Model {model} not available. Run: ollama pull {model}"
                ) from e
            raise
        except TimeoutError:
            return {"type": None, "confidence": 0.0}

        try:
            result = json.loads(body["response"])
            col_type = result.get("type")
            if col_type is not None and col_type not in _VALID_TYPES:
                return {"type": None, "confidence": 0.0}
            return {
                "type": col_type,
                "confidence": float(result.get("confidence", 0.0)),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            if attempt == 0:
                continue
            return {"type": None, "confidence": 0.0}

    return {"type": None, "confidence": 0.0}


def detect_all_columns(
    df: pd.DataFrame,
    model: str = "llama3.1:8b",
    threshold: float = 0.80,
    on_progress=None,
) -> tuple[list[dict], list[dict]]:
    """Classify every column in a DataFrame and split by confidence.

    Args:
        df: Source DataFrame.
        model: Ollama model name.
        threshold: Minimum confidence to include a column.
        on_progress: Optional callback(col_name, status) called before/after
            each column is classified. status is "start" or "done".

    Returns:
        Tuple of (detected, skipped). Each is a list of dicts with keys:
        "column", "type", "confidence".
    """
    detected = []
    skipped = []

    for col in df.columns:
        if on_progress:
            on_progress(col, "start")

        sample = sample_column(df, col)
        result = classify_column(col, sample, model)

        entry = {
            "column": col,
            "type": result["type"],
            "confidence": result["confidence"],
        }

        if result["type"] is not None and result["confidence"] >= threshold:
            detected.append(entry)
        else:
            skipped.append(entry)

        if on_progress:
            on_progress(col, "done")

    return detected, skipped


def build_config(detected: list[dict]) -> dict:
    """Construct a config dict from detected columns.

    Args:
        detected: List of dicts with "column", "type", "confidence" keys.

    Returns:
        Dict mapping column names to their detected type string,
        ready to pass to the anonymizer or save as YAML.
    """
    return {entry["column"]: entry["type"] for entry in detected}
