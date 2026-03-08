import hashlib
from datetime import datetime, timedelta

import pandas as pd


_NAMES = [
    "Alice Martin", "Bob Chen", "Carol Davis", "David Kim", "Eva Rossi",
    "Frank Müller", "Grace Okafor", "Henry Patel", "Iris Johansson", "James Torres",
    "Karen Nguyen", "Liam O'Brien", "Mia Fernandez", "Noah Petrov", "Olivia Yamamoto",
    "Paul Andersen", "Quinn Mwangi", "Rachel Kowalski", "Sam Eriksson", "Tina Dubois",
]

_EMAIL_USERS = [
    "alice", "bob", "carol", "david", "eva", "frank", "grace", "henry",
    "iris", "james", "karen", "liam", "mia", "noah", "olivia", "paul",
    "quinn", "rachel", "sam", "tina", "user", "contact", "info", "hello",
    "admin", "support", "test", "demo", "example", "anon",
]

_EMAIL_DOMAINS = [
    "example.com", "mail.test", "placeholder.org", "fake.net", "anon.io",
    "sample.co", "dummy.com", "noreply.dev", "redacted.net", "obscured.org",
]

_COUNTRIES = [
    "Argentina", "Australia", "Brazil", "Canada", "Chile",
    "Colombia", "Egypt", "France", "Germany", "Ghana",
    "India", "Indonesia", "Italy", "Japan", "Kenya",
    "Mexico", "Netherlands", "Nigeria", "Norway", "Pakistan",
    "Peru", "Philippines", "Poland", "Portugal", "South Africa",
    "South Korea", "Spain", "Sweden", "Thailand", "Turkey",
    "Ukraine", "United Kingdom", "United States", "Vietnam", "Zimbabwe",
]

_DESCRIPTIONS = {
    "Travel": [
        "Sky Airways Int'l", "Horizon Hotels Group", "Metro Transit Authority",
        "Pacific Car Rentals", "Summit Travel Agency", "Coastal Cruise Lines",
        "Atlas Rail Services", "Pioneer Bus Co",
    ],
    "Bills & Utilities": [
        "Greenfield Energy Co", "Clearwater Utilities", "Metro Power Corp",
        "Sunrise Communications", "National Gas & Electric", "Citywide Water Services",
        "BrightNet Internet", "HomeSafe Insurance Co",
    ],
    "Professional Services": [
        "Oakwood Consulting LLC", "Sterling Legal Group", "Apex Accounting Services",
        "Northstar Tax Advisors", "Bridgepoint Financial", "Keystone Business Solutions",
        "Meridian HR Partners", "Pinnacle IT Services",
    ],
    "Food & Drink": [
        "Golden Fork Bistro", "Fresh Harvest Market", "Sunrise Cafe & Bakery",
        "The Corner Deli", "Maple Street Brewery", "Ocean Grill Restaurant",
        "Quick Bites Express", "Garden Valley Grocers",
    ],
    "Shopping": [
        "Marketplace General Store", "Urban Style Outfitters", "Cornerstone Hardware",
        "Elm Street Electronics", "Lakeside Home Goods", "Summit Sports & Outdoors",
        "Birchwood Pharmacy", "Central Office Supplies",
    ],
    "Health & Wellness": [
        "Vitality Health Clinic", "Harmony Wellness Center", "Peak Fitness Studio",
        "Lakeside Medical Group", "Serenity Spa & Body", "ClearView Eye Care",
        "Evergreen Dental Office", "Balanced Life Chiropractic",
    ],
    "Entertainment": [
        "Starlight Cinema", "Echo Music Lounge", "Riverside Arcade & Games",
        "Summit Streaming Media", "Phoenix Concert Hall", "Lakeview Sports Arena",
        "Crescent Theater Co", "Digital Worlds Gaming",
    ],
    "ATM/Cash": [
        "National Bank ATM", "Community Credit Union ATM", "Metro Cash Depot",
        "First Federal ATM", "Citywide Cash Access", "Allied Bank ATM",
    ],
    "Transfer": [
        "Interbank Transfer", "Online Fund Transfer", "Wire Transfer Service",
        "Mobile Pay Transfer", "Direct Deposit Transfer", "ACH Transfer Service",
    ],
    "Default": [
        "General Merchant Co", "Standard Services LLC", "Universal Vendors Inc",
        "Allied Commerce Group", "National Retail Corp", "Metro Business Services",
        "Central Processing Co", "Premier Merchant Services",
    ],
}


def _md5_index(value: str, list_length: int) -> int:
    """Return a stable index into a list by hashing value with MD5.

    Args:
        value: String to hash.
        list_length: Length of the target list.

    Returns:
        Integer in range [0, list_length).
    """
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % list_length


def derive_offset(filename: str) -> int:
    """Derive a deterministic day-offset (180–730) from a filename.

    Args:
        filename: Name of the input file.

    Returns:
        Integer in range [180, 730].
    """
    return 180 + _md5_index(filename + ":offset", 551)


def derive_multiplier(filename: str) -> float:
    """Derive a deterministic multiplier (1.5–3.5) from a filename.

    Args:
        filename: Name of the input file.

    Returns:
        Float in range [1.5, 3.5].
    """
    return 1.5 + _md5_index(filename + ":multiplier", 2001) / 1000


def anonymize_date(value: str, offset: int) -> str:
    """Shift a date forward by offset days, preserving MM/DD/YYYY format.

    Args:
        value: Date string in MM/DD/YYYY format.
        offset: Number of days to shift forward.

    Returns:
        Shifted date string in MM/DD/YYYY format.
    """
    dt = datetime.strptime(value, "%m/%d/%Y")
    return (dt + timedelta(days=offset)).strftime("%m/%d/%Y")


def anonymize_amount(value, multiplier: float):
    """Multiply an amount by multiplier, preserving sign and rounding to 2dp.

    Args:
        value: Numeric amount (string or number).
        multiplier: Scaling factor.

    Returns:
        Scaled amount rounded to 2 decimal places.
    """
    return round(float(value) * multiplier, 2)


def anonymize_description(value: str, category: str) -> str:
    """Replace a description with a deterministic synthetic merchant name.

    Args:
        value: Original description string.
        category: Category to select merchant pool from.

    Returns:
        Synthetic merchant name from the category-appropriate pool.
    """
    pool = _DESCRIPTIONS.get(category, _DESCRIPTIONS["Default"])
    return pool[_md5_index(value, len(pool))]


def anonymize_value(value: str, field_type: str, **kwargs) -> str:
    """Return a deterministic fake replacement for a single PII value.

    The same input always produces the same output, making anonymization
    consistent across runs and datasets.

    Args:
        value: The original PII string to anonymize.
        field_type: One of "name", "email", "country", "date", "amount", or "description".
        **kwargs: Additional parameters for specific field types:
            offset (int): Required for "date" — days to shift forward.
            multiplier (float): Required for "amount" — scaling factor.
            category (str): Required for "description" — merchant category.

    Returns:
        A fake but realistic-looking replacement value.

    Raises:
        ValueError: if field_type is not supported.
    """
    if field_type == "name":
        return _NAMES[_md5_index(value, len(_NAMES))]

    if field_type == "email":
        user = _EMAIL_USERS[_md5_index(value, len(_EMAIL_USERS))]
        # Use a salted hash for the domain so user and domain vary independently
        domain = _EMAIL_DOMAINS[_md5_index(value + ":domain", len(_EMAIL_DOMAINS))]
        return f"{user}@{domain}"

    if field_type == "country":
        return _COUNTRIES[_md5_index(value, len(_COUNTRIES))]

    if field_type == "date":
        return anonymize_date(value, kwargs["offset"])

    if field_type == "amount":
        return anonymize_amount(value, kwargs["multiplier"])

    if field_type == "description":
        return anonymize_description(value, kwargs["category"])

    raise ValueError(f"Unsupported field_type: {field_type!r}")


def anonymize_column(column, field_type, **kwargs):
    """Return a new Series with every value anonymized by field_type.

    Args:
        column: pandas Series of PII strings.
        field_type: One of "name", "email", "country", "date", or "amount".
        **kwargs: Passed through to anonymize_value (e.g. offset, multiplier).

    Returns:
        A new Series with each value replaced by its anonymized equivalent.

    Raises:
        ValueError: if field_type is not supported.
    """
    return column.map(lambda x: anonymize_value(x, field_type, **kwargs) if pd.notna(x) else x)


def anonymize_description_column(description_col, category_col):
    """Return a new Series with descriptions replaced by synthetic merchant names.

    Args:
        description_col: pandas Series of description strings.
        category_col: pandas Series of category strings (same length).

    Returns:
        A new Series with each description replaced by a category-appropriate merchant name.
    """
    return pd.Series(
        [
            anonymize_description(desc, cat) if pd.notna(desc) else desc
            for desc, cat in zip(description_col, category_col)
        ],
        index=description_col.index,
    )


def anonymize_dataframe(df, config, filename=None):
    """Return a copy of df with columns anonymized according to config.

    Args:
        df: pandas DataFrame.
        config: dict mapping column name -> field_type string or dict.
            Simple types: {"email": "email", "full_name": "name"}
            Dict types: {"Description": {"type": "description", "category_column": "Category"}}
        filename: optional input filename used to derive deterministic
            offset (for dates) and multiplier (for amounts).

    Raises:
        KeyError: if a column in config does not exist in df.
        ValueError: if a field_type is not supported.
    """
    missing = [col for col in config if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    offset = derive_offset(filename) if filename else None
    multiplier = derive_multiplier(filename) if filename else None

    result = df.copy()
    for col, field_config in config.items():
        if isinstance(field_config, dict):
            field_type = field_config["type"]
            if field_type == "description":
                cat_col = field_config["category_column"]
                result[col] = anonymize_description_column(result[col], result[cat_col])
            else:
                raise ValueError(f"Unsupported dict field_type: {field_type!r}")
        else:
            kwargs = {}
            if field_config == "date":
                kwargs["offset"] = offset
            elif field_config == "amount":
                kwargs["multiplier"] = multiplier
            elif field_config == "description":
                kwargs["category"] = "Default"
            result[col] = anonymize_column(result[col], field_config, **kwargs)
    return result
