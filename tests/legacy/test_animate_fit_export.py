#!/usr/bin/env python3
"""
Test animation export functionality for the animate_fit module.

This test verifies that animations can be exported reliably whether FFmpeg
is available or not, with automatic fallback to GIF.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cappredict" / "viz"))

from cappredict.viz.animate_fit import (
    generate_synthetic_scurve,
    LogisticFitAnimator,
    save_animation
)


class TestAnimationExport(unittest.TestCase):
    """Test cases for animation export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        self.time_points = 20
        self.frames = 10
        self.fps = 5
        self.dpi = 96
        
    def test_export_with_robust_writer(self):
        """Test that animation export works with robust writer selection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate test data
            t_data, y_data = generate_synthetic_scurve(
                time_points=self.time_points,
                seed=self.seed
            )
            
            # Create animator
            animator = LogisticFitAnimator(t_data, y_data, self.frames, self.fps)
            
            # Export animation
            animation_path = save_animation(
                fig=animator.fig,
                anim=animator.create_animation(),
                outdir=temp_dir,
                basename="test_animation",
                fps=self.fps,
                dpi=self.dpi
            )
            
            # Verify file was created
            self.assertTrue(os.path.exists(animation_path))
            
            # Check file size > 0
            file_size = os.path.getsize(animation_path)
            self.assertGreater(file_size, 0, f"Animation file size is {file_size}, expected > 0")
            
            # Verify file extension is either .mp4 or .gif
            file_ext = Path(animation_path).suffix.lower()
            self.assertIn(file_ext, ['.mp4', '.gif'], f"Unexpected file extension: {file_ext}")
            
            # Verify directory structure
            expected_dir = Path(temp_dir) / f"{Path(animation_path).parent.name}"
            self.assertTrue(expected_dir.exists())
            
            # Verify metrics file was created in same directory
            metrics_path = Path(animation_path).parent / "metrics.json"
            self.assertTrue(metrics_path.exists())
            
    def test_export_directory_creation(self):
        """Test that export creates proper directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate minimal test data
            t_data, y_data = generate_synthetic_scurve(
                time_points=10,
                seed=self.seed
            )
            
            # Create animator
            animator = LogisticFitAnimator(t_data, y_data, frames=3, fps=2)
            
            # Export with custom basename
            animation_path = save_animation(
                fig=animator.fig,
                anim=animator.create_animation(),
                outdir=temp_dir,
                basename="custom_name",
                fps=2,
                dpi=96
            )
            
            # Check directory naming convention
            dir_name = Path(animation_path).parent.name
            self.assertTrue(dir_name.startswith("20"), "Directory should start with year")
            self.assertIn("custom_name", dir_name, "Directory should contain basename")
            
    def test_export_file_naming(self):
        """Test that exported files have correct naming."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate test data
            t_data, y_data = generate_synthetic_scurve(
                time_points=15,
                seed=self.seed
            )
            
            # Create animator
            animator = LogisticFitAnimator(t_data, y_data, frames=5, fps=3)
            
            # Export animation
            animation_path = save_animation(
                fig=animator.fig,
                anim=animator.create_animation(),
                outdir=temp_dir,
                basename="test_export",
                fps=3,
                dpi=96
            )
            
            # Check filename
            filename = Path(animation_path).name
            self.assertTrue(filename.startswith("test_export"), f"Filename should start with basename: {filename}")
            
            # Check extension
            ext = Path(animation_path).suffix.lower()
            self.assertIn(ext, ['.mp4', '.gif'], f"Invalid extension: {ext}")
            
    def test_export_metrics_integration(self):
        """Test that metrics are properly saved alongside animation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate test data
            t_data, y_data = generate_synthetic_scurve(
                time_points=12,
                seed=self.seed
            )
            
            # Create animator
            animator = LogisticFitAnimator(t_data, y_data, frames=4, fps=2)
            
            # Export animation
            animation_path = save_animation(
                fig=animator.fig,
                anim=animator.create_animation(),
                outdir=temp_dir,
                basename="metrics_test",
                fps=2,
                dpi=96
            )
            
            # Verify metrics file exists
            metrics_path = Path(animation_path).parent / "metrics.json"
            self.assertTrue(metrics_path.exists(), "Metrics file should be created")
            
            # Verify metrics file has content
            metrics_size = os.path.getsize(metrics_path)
            self.assertGreater(metrics_size, 0, "Metrics file should not be empty")
            
    def test_export_performance_settings(self):
        """Test that export works with different performance settings."""
        test_cases = [
            (5, 96),   # Low fps, low dpi
            (10, 120), # Medium fps, medium dpi
            (24, 150), # High fps, high dpi
        ]
        
        for fps, dpi in test_cases:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate test data
                t_data, y_data = generate_synthetic_scurve(
                    time_points=8,
                    seed=self.seed
                )
                
                # Create animator
                animator = LogisticFitAnimator(t_data, y_data, frames=3, fps=fps)
                
                # Export animation
                animation_path = save_animation(
                    fig=animator.fig,
                    anim=animator.create_animation(),
                    outdir=temp_dir,
                    basename=f"perf_test_{fps}_{dpi}",
                    fps=fps,
                    dpi=dpi
                )
                
                # Verify export succeeded
                self.assertTrue(os.path.exists(animation_path))
                self.assertGreater(os.path.getsize(animation_path), 0)


if __name__ == '__main__':
    # Set up test environment
    print("🧪 Running animation export tests...")
    
    # Run tests with verbose output
    unittest.main(verbosity=2, exit=False)
    
    print("\n✅ Animation export test suite completed!")
