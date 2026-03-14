# data-cloak

Deterministic data anonymization for CSV files. Replaces real PII — names, emails, countries, dates, amounts, and transaction descriptions — with consistent fake substitutes. The same input always produces the same output.

## What's New in v1.1

**Automatic column detection via local LLM.** When no `--config` is provided, DataCloak samples the CSV, classifies each column using Ollama, and proposes a config for confirmation before anonymizing. The `--config` flag still works exactly as before.

## Usage

```bash
# Auto-detect columns (requires Ollama running locally)
data-cloak transactions.csv

# Use an explicit config (v1.0 behavior, no Ollama needed)
data-cloak transactions.csv --config my_config.toml

# Use a different Ollama model
data-cloak transactions.csv --model mistral
```

Output is written to `<filename>_ANON.csv` in the same directory as the input.

When using auto-detection, a `datacloak_config.yaml` is also saved alongside the output. This file can be reused with `--config` in future runs.

### Auto-Detection Flow

```
$ data-cloak transactions.csv

  Classifying customer_name... done
  Classifying email... done
  Classifying amount... done

DataCloak v1.1 — Auto-detected config for: transactions.csv
────────────────────────────────────────────────────────────

  WILL ANONYMIZE (3 columns):
  ✓  customer_name     →  name          (confidence: 0.97)
  ✓  email             →  email         (confidence: 0.99)
  ✓  amount            →  amount        (confidence: 0.88)

  SKIPPED (low confidence):
  ✗  internal_code     →  ???           (confidence: 0.41)

  [A]ccept  [E]dit config before running  [Q]uit
  >
```

## Configuration

Edit `config.toml` to map CSV column names to field types:

```toml
[columns]
full_name = "name"
email = "email"
country = "country"
```

For financial data, dates and amounts are shifted/scaled deterministically based on the input filename, and descriptions are replaced with synthetic merchant names:

```toml
[columns]
"Transaction Date" = "date"
"Amount" = "amount"
"Description" = {type = "description", category_column = "Category"}
```

Supported field types:

| Type | Behavior |
|---|---|
| `name` | Replaces with a fake name from a fixed pool |
| `email` | Replaces user and domain independently |
| `country` | Replaces with a fake country |
| `date` | Shifts forward by a filename-derived offset (180–730 days) |
| `amount` | Scales by a filename-derived multiplier (1.5–3.5x), preserves sign |
| `description` | Replaces with a synthetic merchant name matched to a category column |
| `id` | Detected by auto-classification (anonymization support planned) |

Config files can be TOML (`.toml`) or YAML (`.yaml`/`.yml`).

## Setup

```bash
uv sync --all-extras
```

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

For auto-detection, [Ollama](https://ollama.com) must be running locally with a model pulled:

```bash
ollama pull llama3.1:8b
```

### Global Install

```bash
uv tool install -e /path/to/data-cloak
```

Then `data-cloak` is available from anywhere.

## Testing

```bash
uv run pytest tests/
```

159 tests covering determinism, distribution, collisions, encoding, null handling, financial transformations, column detection, and CLI integration.
