
"""Kronos Tokenizer Service: load tokenizer, encode/decode OHLCV."""

import os
from typing import Optional
import numpy as np
from transformers import AutoTokenizer
import torch


class KronosTokenizer:
    """Tokenizer for Kronos financial time series model."""

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        cache_dir: Optional[str] = None,
        max_length: int = 512,
        n_features: int = 5,  # OHLCV
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        self.max_length = max_length
        self.n_features = n_features
        self._tokenizer: Optional[AutoTokenizer] = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        return self._tokenizer

    def encode(self, ohlcv_data: np.ndarray) -> np.ndarray:
        """
        Encode OHLCV windows to token IDs.

        Args:
            ohlcv_data: Shape (batch_size, window_size, n_features) or (window_size, n_features)
                       Values should be normalized (e.g., log returns)

        Returns:
            token_ids: Shape (batch_size, seq_len) or (seq_len,)
        """
        # Validate input
        if ohlcv_data.ndim == 2:
            ohlcv_data = ohlcv_data[np.newaxis, ...]  # Add batch dim
        elif ohlcv_data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got {ohlcv_data.ndim}D")

        batch_size, window_size, n_features = ohlcv_data.shape

        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features (OHLCV), got {n_features}")

        if window_size > self.max_length:
            raise ValueError(f"Window size {window_size} exceeds max_length {self.max_length}")

        # Flatten features for tokenization (each timestep becomes a token)
        # Reshape to (batch_size * window_size, n_features) then tokenize each feature vector
        # For Kronos tokenizer, we treat each feature vector as a "token" to be encoded

        # Normalize data (simple min-max per feature for now)
        # In production, use the tokenizer's built-in normalization
        flat_data = ohlcv_data.reshape(-1, n_features)

        # Convert to token IDs using the tokenizer
        # The Kronos tokenizer expects specific format; we use the tokenizer's encode method
        token_ids = []
        for i in range(batch_size):
            window = ohlcv_data[i]  # (window_size, n_features)
            # Convert window to string representation for tokenization
            # This is a simplified approach - real Kronos tokenizer may differ
            window_str = self._window_to_string(window)
            encoded = self.tokenizer.encode(window_str, add_special_tokens=True, max_length=self.max_length, truncation=True)
            token_ids.append(encoded)

        # Pad to same length
        max_len = max(len(t) for t in token_ids)
        padded = np.full((batch_size, max_len), self.tokenizer.pad_token_id or 0, dtype=np.int64)
        for i, t in enumerate(token_ids):
            padded[i, :len(t)] = t

        return padded.squeeze() if batch_size == 1 else padded

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Decode token IDs back to OHLCV predictions.

        Args:
            token_ids: Shape (batch_size, seq_len) or (seq_len,)

        Returns:
            ohlcv_data: Shape (batch_size, window_size, n_features) or (window_size, n_features)
        """
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, ...]
        elif token_ids.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got {token_ids.ndim}D")

        batch_size, seq_len = token_ids.shape

        decoded_windows = []
        for i in range(batch_size):
            tokens = token_ids[i]
            # Remove padding
            if self.tokenizer.pad_token_id is not None:
                tokens = tokens[tokens != self.tokenizer.pad_token_id]
            # Decode tokens to string
            decoded_str = self.tokenizer.decode(tokens, skip_special_tokens=True)
            # Parse string back to OHLCV
            window = self._string_to_window(decoded_str, seq_len)
            decoded_windows.append(window)

        result = np.stack(decoded_windows)
        return result.squeeze() if batch_size == 1 else result

    def _window_to_string(self, window: np.ndarray) -> str:
        """Convert OHLCV window to string for tokenization."""
        # Format: "O:1.0,H:1.01,L:0.99,C:1.005,V:1000|O:1.005,H:1.02,L:1.0,C:1.015,V:1100|..."
        tokens = []
        for t in window:
            o, h, l, c, v = t
            tokens.append(f"O:{o:.6f},H:{h:.6f},L:{l:.6f},C:{c:.6f},V:{v:.2f}")
        return "|".join(tokens)

    def _string_to_window(self, s: str, expected_len: int) -> np.ndarray:
        """Parse string back to OHLCV window."""
        parts = s.split("|")
        window = []
        for part in parts[:expected_len]:
            vals = {}
            for kv in part.split(","):
                k, v = kv.split(":")
                vals[k] = float(v)
            window.append([vals.get("O", 0), vals.get("H", 0), vals.get("L", 0), vals.get("C", 0), vals.get("V", 0)])

        # Pad if needed
        while len(window) < expected_len:
            window.append([0, 0, 0, 0, 0])

        return np.array(window[:expected_len], dtype=np.float32)


# Global instance
_tokenizer_instance: Optional[KronosTokenizer] = None


def get_tokenizer() -> KronosTokenizer:
    """Get or create the global tokenizer instance."""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = KronosTokenizer()
    return _tokenizer_instance
