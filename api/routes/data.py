from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.data_service import fetch_component_data, get_fed_funds_rate

router = APIRouter(prefix="/api/data", tags=["data"])


class PricesRequest(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str


@router.post("/prices")
def get_prices(req: PricesRequest) -> dict:
    """Fetch component price data for given tickers."""
    try:
        df = fetch_component_data(req.tickers, req.start_date, req.end_date)
        if df.empty:
            return {"prices_json": None}
        return {"prices_json": df.to_json(orient="split", date_format="iso")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fed-funds")
def get_fed_funds() -> dict:
    """Fetch historical Fed Funds rate."""
    try:
        series = get_fed_funds_rate()
        if series is None:
            return {"fed_funds_json": None}
        return {"fed_funds_json": series.to_json(orient="split", date_format="iso")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
