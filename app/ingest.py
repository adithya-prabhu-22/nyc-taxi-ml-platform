from pathlib import Path
from datetime import datetime, timezone
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/api")


RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class IngestRequest(BaseModel):
    records: list[dict]


@router.post("/ingest")
def ingest_data(request: IngestRequest):

    if not request.records:
        raise HTTPException(
            status_code=400,
            detail="No records received."
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    file_path = RAW_DATA_DIR / f"batch_{timestamp}.json"

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            request.records,
            file
        )

    return {
        "status": "accepted",
        "records_received": len(request.records),
        "file": str(file_path)
    }