"""
Tests for the animate_fit module.

Tests the logistic curve fitting animation functionality, including
reproducibility and file generation.
"""

import os
import json
import tempfile
import shutil
from unittest.mock import patch
import pytest
import numpy as np

# Import the module to test
from cappredict.viz.animate_fit import (
    logistic_function,
    generate_synthetic_curve,
    fit_logistic_curve,
    calculate_rmse,
    calculate_t90,
    create_fit_animation
)


class TestLogisticFunction:
    """Test the logistic function implementation."""
    
    def test_logistic_function_basic(self):
        """Test basic logistic function calculation."""
        t = np.array([0, 1, 2, 3, 4, 5])
        L, k, t0 = 1.0, 1.0, 2.5
        
        result = logistic_function(t, L, k, t0)
        
        # Check output shape
        assert result.shape == t.shape
        
        # Check that values are in [0, L] range
        assert np.all(result >= 0)
        assert np.all(result <= L)
        
        # Check that inflection point is at t0
        inflection_idx = np.argmax(np.gradient(result))
        assert abs(t[inflection_idx] - t0) < 0.1
    
    def test_logistic_function_parameters(self):
        """Test logistic function with different parameters."""
        t = np.linspace(0, 10, 100)
        
        # Test with different L values
        result1 = logistic_function(t, L=2.0, k=1.0, t0=5.0)
        result2 = logistic_function(t, L=1.0, k=1.0, t0=5.0)
        
        assert np.all(result1 == 2.0 * result2)
        
        # Test with different k values (steepness)
        result3 = logistic_function(t, L=1.0, k=2.0, t0=5.0)
        result4 = logistic_function(t, L=1.0, k=0.5, t0=5.0)
        
        # Higher k should give steeper curve
        assert np.std(result3) > np.std(result4)


class TestSyntheticCurveGeneration:
    """Test synthetic curve generation."""
    
    def test_generate_synthetic_curve_basic(self):
        """Test basic synthetic curve generation."""
        t, y = generate_synthetic_curve(
            time_points=50,
            time_range=(0, 10),
            L=1.0,
            k=1.0,
            t0=5.0,
            noise_level=0.01,
            seed=42
        )
        
        assert len(t) == 50
        assert len(y) == 50
        assert t[0] == 0
        assert t[-1] == 10
        assert np.all(y >= 0)
        assert np.all(y <= 1.0)
    
    def test_generate_synthetic_curve_reproducibility(self):
        """Test that same seed gives reproducible results."""
        seed = 123
        
        t1, y1 = generate_synthetic_curve(seed=seed)
        t2, y2 = generate_synthetic_curve(seed=seed)
        
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_generate_synthetic_curve_different_seeds(self):
        """Test that different seeds give different results."""
        t1, y1 = generate_synthetic_curve(seed=42)
        t2, y2 = generate_synthetic_curve(seed=43)
        
        # Time arrays should be identical
        np.testing.assert_array_equal(t1, t2)
        
        # Value arrays should be different (due to noise)
        assert not np.array_equal(y1, y2)


class TestCurveFitting:
    """Test logistic curve fitting functionality."""
    
    def test_fit_logistic_curve_perfect_data(self):
        """Test fitting with perfect (noiseless) data."""
        t = np.linspace(0, 10, 100)
        L_true, k_true, t0_true = 1.0, 1.5, 5.0
        y_true = logistic_function(t, L_true, k_true, t0_true)
        
        popt, pcov = fit_logistic_curve(t, y_true)
        L_fit, k_fit, t0_fit = popt
        
        # Fitted parameters should be close to true parameters
        assert abs(L_fit - L_true) < 0.01
        assert abs(k_fit - k_true) < 0.01
        assert abs(t0_fit - t0_true) < 0.01
    
    def test_fit_logistic_curve_noisy_data(self):
        """Test fitting with noisy data."""
        t = np.linspace(0, 10, 100)
        L_true, k_true, t0_true = 1.0, 1.5, 5.0
        y_true = logistic_function(t, L_true, k_true, t0_true)
        
        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, len(t))
        y_noisy = np.clip(y_true + noise, 0, 1)
        
        popt, pcov = fit_logistic_curve(t, y_noisy)
        L_fit, k_fit, t0_fit = popt
        
        # Fitted parameters should be reasonably close
        assert abs(L_fit - L_true) < 0.1
        assert abs(k_fit - k_true) < 0.2
        assert abs(t0_fit - t0_true) < 0.2
    
    def test_fit_logistic_curve_insufficient_points(self):
        """Test fitting with insufficient data points."""
        t = np.array([0, 1])
        y = np.array([0.1, 0.2])
        
        # Should handle gracefully and return initial guess
        popt, pcov = fit_logistic_curve(t, y)
        assert len(popt) == 3


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_rmse(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        rmse = calculate_rmse(y_true, y_pred)
        
        # RMSE should be positive
        assert rmse > 0
        
        # RMSE should be reasonable for the given error
        assert 0.1 < rmse < 0.2
    
    def test_calculate_t90(self):
        """Test t90 calculation."""
        L, k, t0 = 1.0, 1.0, 5.0
        t90 = calculate_t90(L, k, t0)
        
        # t90 should be greater than t0
        assert t90 > t0
        
        # For k=1, t90 should be t0 + ln(9) ≈ t0 + 2.197
        expected_t90 = t0 + np.log(9)
        assert abs(t90 - expected_t90) < 0.001


class TestAnimationCreation:
    """Test animation creation functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('matplotlib.animation.FFMpegWriter')
    @patch('matplotlib.animation.FuncAnimation')
    def test_create_fit_animation_basic(self, mock_animation, mock_writer, temp_output_dir):
        """Test basic animation creation."""
        # Mock the animation and writer
        mock_anim = mock_animation.return_value
        mock_writer_instance = mock_writer.return_value
        
        metrics = create_fit_animation(
            seed=42,
            frames=10,
            fps=24,
            output_dir=temp_output_dir,
            show_plot=False
        )
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert 'seed' in metrics
        assert 'final_rmse' in metrics
        assert 'output_files' in metrics
        
        # Check that animation was created
        mock_animation.assert_called_once()
        mock_anim.save.assert_called_once()
    
    def test_create_fit_animation_reproducibility(self, temp_output_dir):
        """Test that same seed gives reproducible results."""
        seed = 42
        
        # Create two animations with same seed
        metrics1 = create_fit_animation(
            seed=seed,
            frames=10,
            fps=24,
            output_dir=temp_output_dir,
            show_plot=False
        )
        
        metrics2 = create_fit_animation(
            seed=seed,
            frames=10,
            fps=24,
            output_dir=temp_output_dir,
            show_plot=False
        )
        
        # Final parameters should be identical
        assert metrics1['final_parameters'] == metrics2['final_parameters']
        assert abs(metrics1['final_rmse'] - metrics2['final_rmse']) < 1e-10
    
    def test_create_fit_animation_different_seeds(self, temp_output_dir):
        """Test that different seeds give different results."""
        metrics1 = create_fit_animation(
            seed=42,
            frames=10,
            fps=24,
            output_dir=temp_output_dir,
            show_plot=False
        )
        
        metrics2 = create_fit_animation(
            seed=43,
            frames=10,
            fps=24,
            output_dir=temp_output_dir,
            show_plot=False
        )
        
        # Parameters should be different
        assert metrics1['final_parameters'] != metrics2['final_parameters']
    
    def test_create_fit_animation_output_files(self, temp_output_dir):
        """Test that output files are created."""
        metrics = create_fit_animation(
            seed=42,
            frames=10,
            fps=24,
            output_dir=temp_output_dir,
            show_plot=False
        )
        
        # Check that output directory was created
        output_path = metrics['output_files']['mp4']
        output_dir = os.path.dirname(output_path)
        assert os.path.exists(output_dir)
        
        # Check that metrics file was created
        metrics_file = metrics['output_files']['metrics']
        assert os.path.exists(metrics_file)
        
        # Check metrics file content
        with open(metrics_file, 'r') as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics['seed'] == 42
        assert saved_metrics['frames'] == 10
        assert saved_metrics['fps'] == 24


if __name__ == "__main__":
    pytest.main([__file__])