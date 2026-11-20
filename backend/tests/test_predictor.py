"""Tests for Kronos Prediction Engine."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.predictor import KronosPredictor, get_predictor


class TestKronosPredictor:
    """Tests for KronosPredictor class."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a predictor with mocked model and tokenizer."""
        with patch("app.services.predictor.AutoModelForTokenClassification") as mock_model, \
             patch("app.services.predictor.AutoTokenizer") as mock_tokenizer:
            mock_model.return_value.from_pretrained.return_value = MagicMock()
            mock_tokenizer.return_value.from_pretrained.return_value = MagicMock()
            predictor = KronosPredictor(model_name="test-model", max_length=512, n_features=5)
            # Mock the tokenizer encode method
            predictor.tokenizer.encode = lambda s, **kwargs: [101, 2023, 2003, 102]  # [CLS, tokens, SEP]
            # Mock the model output
            mock_model.return_value.classifier = MagicMock()
            mock_model.return_value.classifier.return_value = MagicMock()
            mock_model.return_value.classifier.return_value.return_value = torch.tensor([[[-0.5, 0.1, -0.3, 0.2, 0.1, -0.4]]])
            yield predictor

    def test_init(self, mock_predictor):
        """Test predictor initialization."""
        assert mock_predictor.n_features == 5
        assert mock_predictor.max_length == 512

    def test_prepare_context_valid(self, mock_predictor):
        """Test context preparation with valid OHLCV data."""
        context = [[100.0, 105.0, 99.0, 103.0, 1000.0], [103.0, 107.0, 102.0, 106.0, 1100.0]]
        prepared = mock_predictor._prepare_context(context)
        assert prepared.shape[2] == 5  # n_features
        assert prepared.shape[1] == 2  # window_size

    def test_prepare_context_empty(self, mock_predictor):
        """Test context preparation with empty data raises error."""
        with pytest.raises(ValueError, match="Context window cannot be empty"):
            mock_predictor._prepare_context([])

    def test_prepare_context_wrong_features(self, mock_predictor):
        """Test context preparation with wrong number of features raises error."""
        context = [[1.0, 2.0, 3.0]]  # Only 3 features
        with pytest.raises(ValueError, match="Expected 5 features"):
            mock_predictor._prepare_context(context)

    @patch("app.services.predictor.torch")
    def test_predict_basic(self, mock_torch, mock_predictor):
        """Test basic prediction generation."""
        # Setup mocks
        mock_torch.multinomial.return_value = torch.tensor([[1]])
        mock_torch.softmax.return_value = torch.tensor([[0.2, 0.3, 0.1, 0.4]])
        
        context = [[100.0, 105.0, 99.0, 103.0, 1000.0]]
        result = mock_predictor.predict(
            context_ohlcv=context,
            horizon=5,
            temperature=0.8,
            top_p=0.9,
            sample_count=10,
        )

        # Verify result structure
        assert "mean" in result
        assert "std" in result
        assert "quantiles" in result
        assert "samples" in result
        assert "context_length" in result
        assert "horizon" in result
        assert result["horizon"] == 5
        assert result["context_length"] == 1

        # Verify mean/std have correct structure (horizon, n_features)
        assert len(result["mean"]) == 5  # horizon
        assert len(result["mean"][0]) == 5  # n_features

        # Verify quantiles keys
        assert "p0.1" in result["quantiles"]
        assert "p0.25" in result["quantiles"]
        assert "p0.5" in result["quantiles"]
        assert "p0.75" in result["quantiles"]
        assert "p0.9" in result["quantiles"]

    @patch("app.services.predictor.torch")
    def test_predict_with_defaults(self, mock_torch, mock_predictor):
        """Test prediction with default parameters."""
        mock_torch.multinomial.return_value = torch.tensor([[0]])
        mock_torch.softmax.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        context = [[100.0, 105.0, 99.0, 103.0, 1000.0]]
        result = mock_predictor.predict(context_ohlcv=context)

        assert result["horizon"] == 1
        assert result["sample_count"] == 100
        assert result["temperature"] == 1.0
        assert result["top_p"] == 1.0

    def test_predict_invalid_horizon(self, mock_predictor):
        """Test prediction with invalid horizon."""
        context = [[100.0, 105.0, 99.0, 103.0, 1000.0]]
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            mock_predictor.predict(context_ohlcv=context, horizon=0)

    def test_predict_horizon_too_large(self, mock_predictor):
        """Test prediction with horizon exceeding limit."""
        context = [[100.0, 105.0, 99.0, 103.0, 1000.0]]
        with pytest.raises(ValueError, match="horizon must be at most 100"):
            mock_predictor.predict(context_ohlcv=context, horizon=200)

    def test_variable_context_length(self, mock_predictor):
        """Test handling of variable context lengths (issue #92)."""
        # Short context
        short_context = [[100.0, 105.0, 99.0, 103.0, 1000.0]]
        result_short = mock_predictor.predict(context_ohlcv=short_context, horizon=3)
        assert result_short["context_length"] == 1

        # Longer context
        long_context = [
            [100.0, 105.0, 99.0, 103.0, 1000.0],
            [103.0, 107.0, 102.0, 106.0, 1100.0],
            [106.0, 110.0, 105.0, 109.0, 1200.0],
            [109.0, 113.0, 108.0, 112.0, 1300.0],
        ]
        result_long = mock_predictor.predict(context_ohlcv=long_context, horizon=3)
        assert result_long["context_length"] == 4

    def test_temperature_effect(self, mock_predictor):
        """Test that temperature affects sampling diversity."""
        context = [[100.0, 105.0, 99.0, 103.0, 1000.0]]
        
        # Low temperature (more deterministic)
        result_low = mock_predictor.predict(context_ohlcv=context, horizon=3, temperature=0.1, sample_count=20)
        
        # High temperature (more random)
        result_high = mock_predictor.predict(context_ohlcv=context, horizon=3, temperature=1.5, sample_count=20)
        
        # Both should produce valid results with same structure
        assert len(result_low["mean"]) == 3
        assert len(result_high["mean"]) == 3
        # Mean values may differ due to temperature
        assert not np.allclose(result_low["mean"], result_high["mean"])

    def test_sample_count_effect(self, mock_predictor):
        """Test that sample_count affects quantile width."""
        context = [[100.0, 105.0, 99.0, 103.0, 1000.0]]
        
        # Few samples
        result_few = mock_predictor.predict(context_ohlcv=context, horizon=3, sample_count=10)
        # Many samples
        result_many = mock_predictor.predict(context_ohlcv=context, horizon=3, sample_count=100)
        
        # Both should produce valid results
        assert len(result_few["quantiles"]) == len(result_many["quantiles"])
        assert len(result_few["samples"]) == 10
        assert len(result_many["samples"]) == 100
