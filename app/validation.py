import pandas as pd


def validate_data(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "PULocationID",
        "DOLocationID"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"],
        format="mixed",
        errors="coerce"
    )

    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"],
        format="mixed",
        errors="coerce"
    )

    df["trip_distance"] = pd.to_numeric(
        df["trip_distance"],
        errors="coerce"
    )

    df["PULocationID"] = pd.to_numeric(
        df["PULocationID"],
        errors="coerce"
    )

    df["DOLocationID"] = pd.to_numeric(
        df["DOLocationID"],
        errors="coerce"
    )

    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"]
        - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    valid = (
        df["tpep_pickup_datetime"].notna()
        & df["tpep_dropoff_datetime"].notna()
        & df["trip_duration_minutes"].notna()
        & (df["trip_duration_minutes"] > 0)
        & (df["trip_duration_minutes"] <= 180)
        & df["trip_distance"].notna()
        & (df["trip_distance"] > 0)
        & df["PULocationID"].notna()
        & df["DOLocationID"].notna()
    )

    valid_data = df[valid].copy()
    invalid_data = df[~valid].copy()

    return valid_data, invalid_data