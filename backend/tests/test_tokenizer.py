
"""Tests for Kronos Tokenizer Service."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.tokenizer import KronosTokenizer, get_tokenizer


class TestKronosTokenizer:
    """Tests for KronosTokenizer class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a tokenizer with mocked HF tokenizer."""
        with patch("app.services.tokenizer.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_hf_tokenizer = MagicMock()
            mock_hf_tokenizer.pad_token_id = 0
            mock_hf_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_hf_tokenizer.decode.return_value = "O:1.000000,H:1.010000,L:0.990000,C:1.005000,V:1000.00|O:1.005000,H:1.020000,L:1.000000,C:1.015000,V:1100.00"
            mock_from_pretrained.return_value = mock_hf_tokenizer

            tokenizer = KronosTokenizer(model_name="test-model", max_length=100, n_features=5)
            yield tokenizer

    def test_init(self, mock_tokenizer):
        """Test tokenizer initialization."""
        assert mock_tokenizer.model_name == "test-model"
        assert mock_tokenizer.max_length == 100
        assert mock_tokenizer.n_features == 5

    def test_encode_2d_input(self, mock_tokenizer):
        """Test encoding 2D OHLCV input (single window)."""
        # Create sample OHLCV data: (window_size=10, n_features=5)
        ohlcv = np.random.randn(10, 5).astype(np.float32)
        # Make values positive for OHLCV
        ohlcv = np.abs(ohlcv) + 0.1

        token_ids = mock_tokenizer.encode(ohlcv)

        # Should return 2D array (1, seq_len) after squeeze
        assert token_ids.ndim == 1  # squeezed for batch_size=1
        assert len(token_ids) > 0

    def test_encode_3d_input(self, mock_tokenizer):
        """Test encoding 3D OHLCV input (batch)."""
        # Create sample OHLCV data: (batch_size=2, window_size=10, n_features=5)
        ohlcv = np.random.randn(2, 10, 5).astype(np.float32)
        ohlcv = np.abs(ohlcv) + 0.1

        token_ids = mock_tokenizer.encode(ohlcv)

        # Should return 2D array (batch_size, seq_len)
        assert token_ids.ndim == 2
        assert token_ids.shape[0] == 2

    def test_encode_invalid_dimensions(self, mock_tokenizer):
        """Test encoding with invalid input dimensions."""
        # 1D input should fail
        ohlcv_1d = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D input"):
            mock_tokenizer.encode(ohlcv_1d)

        # 4D input should fail
        ohlcv_4d = np.random.randn(1, 2, 10, 5).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D input"):
            mock_tokenizer.encode(ohlcv_4d)

    def test_encode_wrong_feature_count(self, mock_tokenizer):
        """Test encoding with wrong number of features."""
        ohlcv = np.random.randn(10, 3).astype(np.float32)  # Only 3 features
        with pytest.raises(ValueError, match="Expected 5 features"):
            mock_tokenizer.encode(ohlcv)

    def test_encode_window_too_large(self, mock_tokenizer):
        """Test encoding with window larger than max_length."""
        ohlcv = np.random.randn(150, 5).astype(np.float32)  # max_length=100
        with pytest.raises(ValueError, match="exceeds max_length"):
            mock_tokenizer.encode(ohlcv)

    def test_decode_1d_input(self, mock_tokenizer):
        """Test decoding 1D token IDs."""
        token_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        ohlcv = mock_tokenizer.decode(token_ids)

        # Should return 2D array (window_size, n_features) after squeeze
        assert ohlcv.ndim == 2
        assert ohlcv.shape[1] == 5  # 5 features

    def test_decode_2d_input(self, mock_tokenizer):
        """Test decoding 2D token IDs (batch)."""
        token_ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
        ohlcv = mock_tokenizer.decode(token_ids)

        # Should return 3D array (batch_size, window_size, n_features)
        assert ohlcv.ndim == 3
        assert ohlcv.shape[0] == 2
        assert ohlcv.shape[2] == 5

    def test_decode_invalid_dimensions(self, mock_tokenizer):
        """Test decoding with invalid input dimensions."""
        # 3D input should fail
        token_ids_3d = np.random.randint(0, 100, (1, 5, 10)).astype(np.int64)
        with pytest.raises(ValueError, match="Expected 1D or 2D input"):
            mock_tokenizer.decode(token_ids_3d)

    def test_window_to_string(self, mock_tokenizer):
        """Test window to string conversion."""
        window = np.array([
            [1.0, 1.01, 0.99, 1.005, 1000],
            [1.005, 1.02, 1.0, 1.015, 1100],
        ], dtype=np.float32)

        s = mock_tokenizer._window_to_string(window)

        assert "O:1.000000,H:1.010000,L:0.990000,C:1.005000,V:1000.00" in s
        assert "|" in s

    def test_string_to_window(self, mock_tokenizer):
        """Test string to window conversion."""
        s = "O:1.000000,H:1.010000,L:0.990000,C:1.005000,V:1000.00|O:1.005000,H:1.020000,L:1.000000,C:1.015000,V:1100.00"
        window = mock_tokenizer._string_to_window(s, expected_len=2)

        assert window.shape == (2, 5)
        assert window[0, 0] == 1.0  # Open
        assert window[0, 1] == 1.01  # High
        assert window[0, 2] == 0.99  # Low
        assert window[0, 3] == 1.005  # Close
        assert window[0, 4] == 1000.0  # Volume

    def test_string_to_window_padding(self, mock_tokenizer):
        """Test string to window with padding."""
        s = "O:1.000000,H:1.010000,L:0.990000,C:1.005000,V:1000.00"
        window = mock_tokenizer._string_to_window(s, expected_len=3)

        assert window.shape == (3, 5)
        # Last row should be zeros (padded)
        assert np.allclose(window[2], 0)


class TestTokenizerAPI:
    """Tests for tokenizer API endpoints."""

    @pytest.mark.asyncio
    async def test_tokenize_endpoint(self, client):
        """Test /api/v1/tokenizer/tokenize endpoint."""
        # Mock the tokenizer
        with patch("app.api.tokenizer_routes.get_tokenizer") as mock_get_tokenizer:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
            mock_get_tokenizer.return_value = mock_tokenizer

            response = await client.post(
                "/api/v1/tokenizer/tokenize",
                json={"ohlcv": [[[1.0, 1.01, 0.99, 1.005, 1000]] * 10]}
            )

            assert response.status_code == 200
            data = response.json()
            assert "token_ids" in data
            assert "shape" in data
            assert data["shape"] == [1, 5]

    @pytest.mark.asyncio
    async def test_tokenize_endpoint_invalid_input(self, client):
        """Test tokenize endpoint with invalid input."""
        response = await client.post(
            "/api/v1/tokenizer/tokenize",
            json={"ohlcv": [[[1.0, 1.01, 0.99]]]}  # Only 3 features
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_detokenize_endpoint(self, client):
        """Test /api/v1/tokenizer/detokenize endpoint."""
        with patch("app.api.tokenizer_routes.get_tokenizer") as mock_get_tokenizer:
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = np.array([[[1.0, 1.01, 0.99, 1.005, 1000]] * 10], dtype=np.float32)
            mock_get_tokenizer.return_value = mock_tokenizer

            response = await client.post(
                "/api/v1/tokenizer/detokenize",
                json={"token_ids": [[1, 2, 3, 4, 5]]}
            )

            assert response.status_code == 200
            data = response.json()
            assert "ohlcv" in data
            assert "shape" in data

    @pytest.mark.asyncio
    async def test_detokenize_endpoint_invalid_input(self, client):
        """Test detokenize endpoint with invalid input."""
        response = await client.post(
            "/api/v1/tokenizer/detokenize",
            json={"token_ids": [[[1, 2, 3]]]}  # 3D input
        )

        assert response.status_code == 400


class TestGetTokenizer:
    """Tests for get_tokenizer singleton."""

    def test_singleton(self):
        """Test that get_tokenizer returns the same instance."""
        with patch("app.services.tokenizer.AutoTokenizer.from_pretrained"):
            t1 = get_tokenizer()
            t2 = get_tokenizer()
            assert t1 is t2
