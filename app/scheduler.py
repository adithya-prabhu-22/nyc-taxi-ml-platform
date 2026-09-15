import asyncio

from app.pipeline import process_data
from app.window import get_latest_window
from app.train import train_models


RETRAIN_INTERVAL_HOURS = 12


async def run_training_cycle():

    print("Starting training cycle...")

    valid_data, invalid_data = process_data()

    if valid_data.empty:
        print("No valid data available.")
        return

    window_data = get_latest_window(
        valid_data
    )

    if window_data.empty:
        print("No data available for training.")
        return

    print(
        f"Training on {len(window_data)} records."
    )

    results = train_models(
        window_data
    )

    print(
        f"Best model: {results['best_model']}"
    )

    for model_name in [
        "knn",
        "random_forest",
        "xgboost"
    ]:
        metrics = results[model_name]

        print(
            f"{model_name}: "
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"R2={metrics['R2']:.4f}"
        )


async def start_scheduler():

    while True:

        await run_training_cycle()

        print(
            f"Waiting {RETRAIN_INTERVAL_HOURS} hours "
            "until the next training cycle..."
        )

        await asyncio.sleep(
            RETRAIN_INTERVAL_HOURS * 60 * 60
        )