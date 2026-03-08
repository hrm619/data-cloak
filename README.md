# data-cloak

Deterministic data anonymization for CSV files. Replaces real PII — names, emails, countries, dates, amounts, and transaction descriptions — with consistent fake substitutes. The same input always produces the same output.

## Usage

```bash
uv run python main.py users.csv
uv run python main.py users.csv --config my_config.toml
```

Output is written to `<filename>_ANON.csv` in the same directory as the input.

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

## Setup

```bash
uv sync
```

Requires Python 3.13+.
