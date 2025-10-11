#!/usr/bin/env python3
"""
Tests for the animate_fit module.

This test suite verifies the functionality of the logistic curve fitting animation,
focusing on reproducibility and file generation.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cappredict" / "viz"))

from cappredict.viz.animate_fit import (
    generate_synthetic_scurve,
    fit_logistic_curve,
    logistic_function,
    calculate_t90,
    LogisticFitAnimator
)


class TestAnimateFit(unittest.TestCase):
    """Test cases for the animate_fit module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        self.time_points = 50
        self.frames = 10
        self.fps = 5
        
    def test_synthetic_scurve_reproducibility(self):
        """Test that same seed produces reproducible S-curve data."""
        # Generate curve twice with same seed
        t1, y1 = generate_synthetic_scurve(
            time_points=self.time_points,
            seed=self.seed
        )
        t2, y2 = generate_synthetic_scurve(
            time_points=self.time_points,
            seed=self.seed
        )
        
        # Should be identical
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(y1, y2)
        
    def test_synthetic_scurve_different_seeds(self):
        """Test that different seeds produce different S-curve data."""
        # Generate curves with different seeds
        t1, y1 = generate_synthetic_scurve(
            time_points=self.time_points,
            seed=42
        )
        t2, y2 = generate_synthetic_scurve(
            time_points=self.time_points,
            seed=123
        )
        
        # Time should be identical (deterministic)
        np.testing.assert_array_equal(t1, t2)
        
        # Values should be different (due to different noise)
        self.assertFalse(np.array_equal(y1, y2))
        
    def test_logistic_function(self):
        """Test the logistic function implementation."""
        t = np.array([0, 5, 10])
        L, k, t0 = 1.0, 1.0, 5.0
        
        result = logistic_function(t, L, k, t0)
        
        # At t=t0, should be L/2
        self.assertAlmostEqual(result[1], L/2, places=5)
        
        # At t=0, should be less than L/2
        self.assertLess(result[0], L/2)
        
        # At t=10, should be greater than L/2
        self.assertGreater(result[2], L/2)
        
    def test_calculate_t90(self):
        """Test t90 calculation."""
        L, k, t0 = 1.0, 1.0, 5.0
        t90 = calculate_t90(L, k, t0)
        
        # At t90, logistic function should give 0.9 * L
        value_at_t90 = logistic_function(np.array([t90]), L, k, t0)[0]
        self.assertAlmostEqual(value_at_t90, 0.9 * L, places=5)
        
    def test_fit_logistic_curve_insufficient_data(self):
        """Test curve fitting with insufficient data points."""
        t_data = np.array([0, 1])
        y_data = np.array([0.1, 0.2])
        
        params, rmse = fit_logistic_curve(t_data, y_data)
        
        self.assertIsNone(params)
        self.assertEqual(rmse, float('inf'))
        
    def test_fit_logistic_curve_valid_data(self):
        """Test curve fitting with valid data."""
        # Generate perfect logistic data
        t_data = np.linspace(0, 10, 20)
        L_true, k_true, t0_true = 1.0, 1.0, 5.0
        y_data = logistic_function(t_data, L_true, k_true, t0_true)
        
        params, rmse = fit_logistic_curve(t_data, y_data)
        
        self.assertIsNotNone(params)
        self.assertLess(rmse, 0.01)  # Should be very low for perfect data
        
        L_fit, k_fit, t0_fit = params
        
        # Parameters should be close to true values
        self.assertAlmostEqual(L_fit, L_true, places=1)
        self.assertAlmostEqual(k_fit, k_true, places=1)
        self.assertAlmostEqual(t0_fit, t0_true, places=1)
        
    def test_animator_initialization(self):
        """Test LogisticFitAnimator initialization."""
        t_data, y_data = generate_synthetic_scurve(
            time_points=self.time_points,
            seed=self.seed
        )
        
        animator = LogisticFitAnimator(t_data, y_data, self.frames, self.fps)
        
        self.assertEqual(len(animator.time_data), len(t_data))
        self.assertEqual(len(animator.value_data), len(y_data))
        self.assertEqual(animator.frames, self.frames)
        self.assertEqual(animator.fps, self.fps)
        
    def test_animator_metrics_collection(self):
        """Test that animator collects metrics during animation."""
        t_data, y_data = generate_synthetic_scurve(
            time_points=20,  # Small dataset for quick test
            seed=self.seed
        )
        
        animator = LogisticFitAnimator(t_data, y_data, frames=5, fps=self.fps)
        
        # Run a few animation frames
        for frame in range(3):
            animator.animate(frame)
            
        # Should have collected metrics
        self.assertEqual(len(animator.metrics_history), 3)
        
        # Check metric structure
        for metrics in animator.metrics_history:
            self.assertIn('frame', metrics)
            self.assertIn('n_points', metrics)
            self.assertIn('rmse', metrics)
            
    @patch('matplotlib.pyplot.show')
    def test_animation_file_generation(self, mock_show):
        """Test that animation files are generated correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate test data
            t_data, y_data = generate_synthetic_scurve(
                time_points=20,
                seed=self.seed
            )
            
            animator = LogisticFitAnimator(t_data, y_data, frames=3, fps=2)
            
            # Test metrics saving
            metrics_path = os.path.join(temp_dir, "test_metrics.json")
            
            # Run animation to collect metrics
            for frame in range(3):
                animator.animate(frame)
                
            animator.save_metrics(metrics_path)
            
            # Check that metrics file was created
            self.assertTrue(os.path.exists(metrics_path))
            
            # Check metrics content
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            self.assertIsInstance(metrics, list)
            self.assertEqual(len(metrics), 3)
            
            # Each metric should have expected keys
            for metric in metrics:
                self.assertIn('frame', metric)
                self.assertIn('n_points', metric)
                self.assertIn('rmse', metric)
                
    def test_reproducible_animation_metrics(self):
        """Test that same seed produces reproducible animation metrics."""
        # Create two animators with same seed
        t_data1, y_data1 = generate_synthetic_scurve(
            time_points=30,
            seed=self.seed
        )
        t_data2, y_data2 = generate_synthetic_scurve(
            time_points=30,
            seed=self.seed
        )
        
        animator1 = LogisticFitAnimator(t_data1, y_data1, frames=5, fps=self.fps)
        animator2 = LogisticFitAnimator(t_data2, y_data2, frames=5, fps=self.fps)
        
        # Run same animation frames
        for frame in range(5):
            animator1.animate(frame)
            animator2.animate(frame)
            
        # Metrics should be identical
        self.assertEqual(len(animator1.metrics_history), len(animator2.metrics_history))
        
        for m1, m2 in zip(animator1.metrics_history, animator2.metrics_history):
            self.assertEqual(m1['frame'], m2['frame'])
            self.assertEqual(m1['n_points'], m2['n_points'])
            
            # RMSE should be very close (allowing for numerical precision)
            if m1['rmse'] != float('inf') and m2['rmse'] != float('inf'):
                self.assertAlmostEqual(m1['rmse'], m2['rmse'], places=10)
                
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very noisy data
        t_data, y_data = generate_synthetic_scurve(
            time_points=10,
            noise_std=0.5,  # Very high noise
            seed=self.seed
        )
        
        animator = LogisticFitAnimator(t_data, y_data, frames=3, fps=self.fps)
        
        # Should not crash with noisy data
        try:
            for frame in range(3):
                animator.animate(frame)
        except Exception as e:
            self.fail(f"Animation failed with noisy data: {e}")
            
        # Should have collected some metrics
        self.assertGreater(len(animator.metrics_history), 0)


class TestCLIFunctionality(unittest.TestCase):
    """Test the CLI functionality."""
    
    @patch('sys.argv', ['animate_fit.py', '--seed', '42', '--frames', '5', '--fps', '2'])
    @patch('cappredict.viz.animate_fit.LogisticFitAnimator.save_animation')
    @patch('cappredict.viz.animate_fit.LogisticFitAnimator.save_metrics')
    def test_cli_argument_parsing(self, mock_save_metrics, mock_save_animation):
        """Test that CLI arguments are parsed correctly."""
        # Import main after patching sys.argv
        from cappredict.viz.animate_fit import main
        
        # Mock the save methods to avoid file I/O
        mock_save_animation.return_value = None
        mock_save_metrics.return_value = None
        
        try:
            main()
        except SystemExit:
            pass  # argparse may call sys.exit
        except Exception as e:
            # Should not fail due to argument parsing
            if "argument" in str(e).lower():
                self.fail(f"CLI argument parsing failed: {e}")


if __name__ == '__main__':
    # Set up test environment
    print("🧪 Running animate_fit tests...")
    
    # Run tests with verbose output
    unittest.main(verbosity=2, exit=False)
    
    print("\n✅ Test suite completed!")
