import pandas as pd


WINDOW_HOURS = 24


def get_latest_window(
    df: pd.DataFrame,
    hours: int = WINDOW_HOURS
) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    df = df.copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["tpep_pickup_datetime"]
    )

    latest_time = df["tpep_pickup_datetime"].max()

    window_start = latest_time - pd.Timedelta(
        hours=hours
    )

    window_df = df[
        df["tpep_pickup_datetime"] >= window_start
    ].copy()

    return window_df.sort_values(
        "tpep_pickup_datetime"
    ).reset_index(drop=True)