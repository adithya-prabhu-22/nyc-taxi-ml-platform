import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pickup_time = df["tpep_pickup_datetime"]

    df["pickup_hour"] = pickup_time.dt.hour
    df["day_of_week"] = pickup_time.dt.dayofweek
    df["pickup_day"] = pickup_time.dt.day
    df["pickup_month"] = pickup_time.dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_rush_hour"] = (
        pickup_time.dt.hour.isin([7, 8, 9, 16, 17, 18, 19])
    ).astype(int)

    df["route"] = (
        df["PULocationID"].astype(str)
        + "_"
        + df["DOLocationID"].astype(str)
    )

    return df