import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }