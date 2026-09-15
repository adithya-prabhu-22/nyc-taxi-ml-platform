from pathlib import Path
import json

import pandas as pd

from app.validation import validate_data
from app.preprocessing import preprocess_data
from app.features import create_features


RAW_DATA_DIR = Path("data/raw")


def load_raw_batches() -> pd.DataFrame:
    files = sorted(
        RAW_DATA_DIR.glob("*.json")
    )

    if not files:
        return pd.DataFrame()

    batches = []

    for file in files:
        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:
            records = json.load(f)

        if records:
            batches.extend(records)

    if not batches:
        return pd.DataFrame()

    return pd.DataFrame(batches)


def process_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    df = load_raw_batches()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    valid_data, invalid_data = validate_data(df)

    if valid_data.empty:
        return pd.DataFrame(), invalid_data

    valid_data = preprocess_data(
        valid_data
    )

    valid_data = create_features(
        valid_data
    )

    return valid_data, invalid_data