"""Prediction Engine API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.services.predictor import get_predictor, KronosPredictor


router = APIRouter(prefix="/predict", tags=["prediction"])


class PredictRequest(BaseModel):
    """Request for generating predictions with sampling."""

    context_window: List[List[float]] = Field(
        ..., description="OHLCV context window, list of [O, H, L, C, V] values"
    )
    horizon: int = Field(
        1, ge=1, le=100, description="Number of future time steps to predict"
    )
    temperature: float = Field(
        1.0, ge=0.1, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        1.0, ge=0.01, le=1.0, description="Top-p nucleus sampling parameter"
    )
    sample_count: int = Field(
        100, ge=1, le=500, description="Number of samples to draw"
    )


class PredictResponse(BaseModel):
    """Response with sampled predictions and statistics."""

    mean: List[List[float]] = Field(
        ..., description="Mean prediction across samples"
    )
    std: List[List[float]] = Field(
        ..., description="Standard deviation across samples"
    )
    quantiles: Dict[str, List[List[float]]] = Field(
        ..., description="Quantile predictions"
    )
    samples: List[List[List[float]]] = Field(
        ..., description="Full sampled predictions"
    )
    context_length: int = Field(
        ..., description="Original context window length"
    )
    horizon: int = Field(..., description="Prediction horizon")
    temperature: float = Field(..., description="Sampling temperature")
    top_p: float = Field(..., description="Top-p nucleus sampling parameter")
    sample_count: int = Field(..., description="Number of samples drawn")
