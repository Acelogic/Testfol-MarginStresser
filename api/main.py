import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from api.routes import backtest, data

app = FastAPI(title="Testfol Backend", version="1.0.0")

# CORS — allow Streamlit on :8501
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(backtest.router)
app.include_router(data.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
