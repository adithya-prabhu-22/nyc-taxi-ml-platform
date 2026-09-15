import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "payment_type"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["passenger_count"] = df["passenger_count"].fillna(
        df["passenger_count"].median()
    )

    df["RatecodeID"] = df["RatecodeID"].fillna(
        df["RatecodeID"].mode()[0]
    )

    df["payment_type"] = df["payment_type"].fillna(
        df["payment_type"].mode()[0]
    )

    return df