"""
Production FastAPI Inference & Drift Observability Microservice for GKE / Vertex AI Endpoints.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List

app = FastAPI(title="Cloud-to-Edge IoT Telemetry MLOps Service", version="1.0.0")


class TelemetryPayload(BaseModel):
    device_id: str
    sensor_type: str
    temp_c: float
    temp_roll_mean: float
    temp_roll_std: float
    temp_delta_1: float
    temp_zscore: float
    vib_rms: float
    vib_roll_mean: float
    vib_zscore: float
    vib_spectral_energy: float
    power_roll_mean: float
    power_roll_std: float
    latency_roll_mean: float
    latency_roll_std: float
    security_duty_cycle: float


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "healthy", "engine": "vertex-ai-iot-mlops-v1", "quantization": "INT8_EDGE_READY"}
