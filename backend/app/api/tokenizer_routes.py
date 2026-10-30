
"""Tokenizer API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from app.services.tokenizer import get_tokenizer


router = APIRouter(prefix="/tokenizer", tags=["tokenizer"])


class TokenizeRequest(BaseModel):
    """Request for tokenizing OHLCV data."""

    ohlcv: List[List[List[float]]] = Field(
        ...,
        description="OHLCV data with shape (batch_size, window_size, 5) or (window_size, 5)",
        min_length=1,
    )


class TokenizeResponse(BaseModel):
    """Response with token IDs."""

    token_ids: List[List[int]] = Field(..., description="Token IDs with shape (batch_size, seq_len)")
    shape: List[int] = Field(..., description="Shape of token_ids")


class DetokenizeRequest(BaseModel):
    """Request for detokenizing token IDs."""

    token_ids: List[List[int]] = Field(..., description="Token IDs with shape (batch_size, seq_len)", min_length=1)


class DetokenizeResponse(BaseModel):
    """Response with OHLCV predictions."""

    ohlcv: List[List[List[float]]] = Field(..., description="OHLCV data with shape (batch_size, window_size, 5)")
    shape: List[int] = Field(..., description="Shape of OHLCV data")


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """
    Encode OHLCV windows to token IDs.

    Accepts OHLCV data with shape (batch_size, window_size, 5) or (window_size, 5).
    Returns token IDs for use with Kronos model.
    """
    try:
        tokenizer = get_tokenizer()

        # Convert to numpy array
        ohlcv_array = np.array(request.ohlcv, dtype=np.float32)

        # Handle both 2D and 3D input
        if ohlcv_array.ndim == 2:
            ohlcv_array = ohlcv_array[np.newaxis, ...]
        elif ohlcv_array.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got {ohlcv_array.ndim}D")

        # Encode
        token_ids = tokenizer.encode(ohlcv_array)

        # Ensure 2D output
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, ...]

        return TokenizeResponse(
            token_ids=token_ids.tolist(),
            shape=list(token_ids.shape)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")


@router.post("/detokenize", response_model=DetokenizeResponse)
async def detokenize(request: DetokenizeRequest):
    """
    Decode token IDs back to OHLCV predictions.

    Accepts token IDs with shape (batch_size, seq_len).
    Returns OHLCV data with shape (batch_size, window_size, 5).
    """
    try:
        tokenizer = get_tokenizer()

        # Convert to numpy array
        token_ids_array = np.array(request.token_ids, dtype=np.int64)

        # Handle both 1D and 2D input
        if token_ids_array.ndim == 1:
            token_ids_array = token_ids_array[np.newaxis, ...]
        elif token_ids_array.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got {token_ids_array.ndim}D")

        # Decode
        ohlcv = tokenizer.decode(token_ids_array)

        # Ensure 3D output
        if ohlcv.ndim == 2:
            ohlcv = ohlcv[np.newaxis, ...]

        return DetokenizeResponse(
            ohlcv=ohlcv.tolist(),
            shape=list(ohlcv.shape)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detokenization failed: {str(e)}")
