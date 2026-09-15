from pathlib import Path

import joblib
import pandas as pd


PRODUCTION_MODEL_PATH = Path("models/production_model.joblib")


def load_production_model():
    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production model not found."
        )

    return joblib.load(PRODUCTION_MODEL_PATH)


def predict_trip_duration(data: dict) -> float:
    model = load_production_model()

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return float(prediction)