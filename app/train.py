from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTION_MODEL_PATH = MODEL_DIR / "production_model.joblib"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "NYC Taxi Training"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


FEATURE_COLUMNS = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
    "payment_type",
    "pickup_hour",
    "day_of_week",
    "pickup_day",
    "is_weekend",
    "is_rush_hour"
]

TARGET_COLUMN = "trip_duration_minutes"


CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
    "payment_type"
]

NUMERICAL_FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "day_of_week",
    "pickup_day",
    "is_weekend",
    "is_rush_hour"
]


MAX_TRAINING_ROWS = 150000


def build_preprocessor(scale_numeric: bool = False):

    numerical_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES
            )
        ]
    )


def evaluate_predictions(
    y_true: pd.Series,
    predictions
) -> dict:

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = mean_squared_error(
        y_true,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def train_models(df: pd.DataFrame) -> dict:

    df = df.copy()

    df = df.sort_values(
        "tpep_pickup_datetime"
    ).reset_index(drop=True)

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    if len(train_df) > MAX_TRAINING_ROWS:

        train_df = train_df.sample(
            n=MAX_TRAINING_ROWS,
            random_state=42
        )

        train_df = train_df.sort_values(
            "tpep_pickup_datetime"
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    models = {
        "knn": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        scale_numeric=True
                    )
                ),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=10,
                        weights="distance",
                        n_jobs=-1
                    )
                )
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor()
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        n_jobs=-1,
                        random_state=42
                    )
                )
            ]
        ),

        "xgboost": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor()
                ),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=200,
                        max_depth=8,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        )
    }

    results = {}

    for name, pipeline in models.items():

        with mlflow.start_run(
            run_name=f"{name}_training"
        ):

            mlflow.set_tag(
                "model_type",
                name
            )

            mlflow.set_tag(
                "stage",
                "candidate"
            )

            mlflow.log_param(
                "training_rows",
                len(X_train)
            )

            mlflow.log_param(
                "test_rows",
                len(X_test)
            )

            mlflow.log_param(
                "max_training_rows",
                MAX_TRAINING_ROWS
            )

            if name == "knn":

                mlflow.log_param(
                    "n_neighbors",
                    10
                )

                mlflow.log_param(
                    "weights",
                    "distance"
                )

                mlflow.log_param(
                    "scaled_numeric",
                    True
                )

            elif name == "random_forest":

                mlflow.log_param(
                    "n_estimators",
                    100
                )

                mlflow.log_param(
                    "random_state",
                    42
                )

            elif name == "xgboost":

                mlflow.log_param(
                    "n_estimators",
                    200
                )

                mlflow.log_param(
                    "max_depth",
                    8
                )

                mlflow.log_param(
                    "learning_rate",
                    0.1
                )

                mlflow.log_param(
                    "subsample",
                    0.8
                )

                mlflow.log_param(
                    "colsample_bytree",
                    0.8
                )

                mlflow.log_param(
                    "random_state",
                    42
                )

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            metrics = evaluate_predictions(
                y_test,
                predictions
            )

            mlflow.log_metric(
                "MAE",
                metrics["MAE"]
            )

            mlflow.log_metric(
                "RMSE",
                metrics["RMSE"]
            )

            mlflow.log_metric(
                "R2",
                metrics["R2"]
            )

            model_path = MODEL_DIR / f"{name}.joblib"

            joblib.dump(
                pipeline,
                model_path
            )

            mlflow.log_artifact(
                str(model_path),
                artifact_path="models"
            )

            results[name] = {
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "model_path": str(model_path)
            }

    best_model = min(
        results,
        key=lambda name: results[name]["MAE"]
    )

    best_model_path = Path(
        results[best_model]["model_path"]
    )

    candidate_model = joblib.load(
        best_model_path
    )

    candidate_predictions = candidate_model.predict(
        X_test
    )

    candidate_metrics = evaluate_predictions(
        y_test,
        candidate_predictions
    )

    production_exists = PRODUCTION_MODEL_PATH.exists()

    if production_exists:

        production_model = joblib.load(
            PRODUCTION_MODEL_PATH
        )

        production_predictions = production_model.predict(
            X_test
        )

        production_metrics = evaluate_predictions(
            y_test,
            production_predictions
        )

        if candidate_metrics["MAE"] < production_metrics["MAE"]:

            joblib.dump(
                candidate_model,
                PRODUCTION_MODEL_PATH
            )

            production_status = "promoted"
            production_model_name = best_model

        else:

            production_status = "kept"
            production_model_name = "existing"

    else:

        joblib.dump(
            candidate_model,
            PRODUCTION_MODEL_PATH
        )

        production_status = "promoted"
        production_model_name = best_model

        production_metrics = None

    with mlflow.start_run(
        run_name="deployment_decision"
    ):

        mlflow.set_tag(
            "stage",
            "deployment_decision"
        )

        mlflow.set_tag(
            "best_candidate",
            best_model
        )

        mlflow.set_tag(
            "production_status",
            production_status
        )

        mlflow.set_tag(
            "production_model",
            production_model_name
        )

        mlflow.log_metric(
            "candidate_MAE",
            candidate_metrics["MAE"]
        )

        mlflow.log_metric(
            "candidate_RMSE",
            candidate_metrics["RMSE"]
        )

        mlflow.log_metric(
            "candidate_R2",
            candidate_metrics["R2"]
        )

        if production_metrics is not None:

            mlflow.log_metric(
                "production_MAE",
                production_metrics["MAE"]
            )

            mlflow.log_metric(
                "production_RMSE",
                production_metrics["RMSE"]
            )

            mlflow.log_metric(
                "production_R2",
                production_metrics["R2"]
            )

    results["best_model"] = best_model

    results["candidate_metrics"] = candidate_metrics

    results["production_status"] = production_status

    results["production_model"] = production_model_name

    results["production_model_path"] = str(
        PRODUCTION_MODEL_PATH
    )

    results["production_metrics"] = production_metrics

    return results