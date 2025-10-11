#!/usr/bin/env python3
"""
Demo script for the logistic curve fitting animation.

This script demonstrates how to use the animate_fit module.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cappredict.viz.animate_fit import create_fit_animation

def main():
    """Run a demo of the animation."""
    print("🎬 CapPredict Logistic Curve Fitting Animation Demo")
    print("=" * 50)
    
    try:
        # Create a short animation for demo purposes
        print("Creating animation with 30 frames...")
        metrics = create_fit_animation(
            seed=42,
            frames=30,  # Short for demo
            fps=10,     # Slower for demo
            output_dir="artifacts",
            show_plot=False
        )
        
        print("\n✅ Animation created successfully!")
        print(f"   MP4 saved to: {metrics['output_files']['mp4']}")
        print(f"   Metrics saved to: {metrics['output_files']['metrics']}")
        print(f"   Final RMSE: {metrics['final_rmse']:.4f}")
        print(f"   Correlation: {metrics['correlation']:.4f}")
        print(f"   Final parameters:")
        print(f"     L (max): {metrics['final_parameters']['L']:.3f}")
        print(f"     k (rate): {metrics['final_parameters']['k']:.3f}")
        print(f"     t0 (inflection): {metrics['final_parameters']['t0']:.3f}")
        print(f"     t90 (90% completion): {metrics['final_parameters']['t90']:.3f}")
        
    except Exception as e:
        print(f"❌ Error creating animation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())