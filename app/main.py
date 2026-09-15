from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ingest import router as ingest_router
from app.predict import predict_trip_duration


app = FastAPI(
    title="NYC Taxi ML Platform",
    version="1.0.0"
)


app.include_router(ingest_router)


class PredictionRequest(BaseModel):
    trip_distance: float
    passenger_count: float
    PULocationID: int
    DOLocationID: int
    RatecodeID: int
    payment_type: int
    pickup_hour: int
    day_of_week: int
    pickup_day: int
    is_weekend: int
    is_rush_hour: int


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/api/predict")
def predict(request: PredictionRequest):

    try:
        prediction = predict_trip_duration(
            request.model_dump()
        )

        return {
            "predicted_trip_duration_minutes": prediction
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )