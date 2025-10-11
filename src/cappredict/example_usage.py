#!/usr/bin/env python3
"""
Example usage of the CapPredict animation module.

This script demonstrates various ways to use the logistic curve fitting animation.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cappredict.viz.animate_fit import create_fit_animation

def example_basic_usage():
    """Basic usage example."""
    print("🎬 Example 1: Basic Usage")
    print("-" * 30)
    
    metrics = create_fit_animation(
        seed=42,
        frames=50,  # Quick animation
        fps=10,
        output_dir="examples/basic",
        show_plot=False
    )
    
    print(f"✅ Animation created with {metrics['frames']} frames")
    print(f"   Final RMSE: {metrics['final_rmse']:.4f}")
    print(f"   Saved to: {metrics['output_files']['mp4']}")
    print()

def example_high_quality():
    """High-quality animation for presentations."""
    print("🎬 Example 2: High-Quality Animation")
    print("-" * 30)
    
    metrics = create_fit_animation(
        seed=123,
        frames=200,  # Smooth animation
        fps=30,      # Professional frame rate
        output_dir="examples/high_quality",
        show_plot=False
    )
    
    print(f"✅ High-quality animation created with {metrics['frames']} frames")
    print(f"   Final RMSE: {metrics['final_rmse']:.4f}")
    print(f"   Saved to: {metrics['output_files']['mp4']}")
    print()

def example_reproducibility():
    """Demonstrate reproducibility with same seed."""
    print("🎬 Example 3: Reproducibility Test")
    print("-" * 30)
    
    seed = 999
    
    # Create two animations with same seed
    metrics1 = create_fit_animation(
        seed=seed,
        frames=30,
        fps=10,
        output_dir="examples/reproducibility/run1",
        show_plot=False
    )
    
    metrics2 = create_fit_animation(
        seed=seed,
        frames=30,
        fps=10,
        output_dir="examples/reproducibility/run2",
        show_plot=False
    )
    
    # Check if results are identical
    rmse_match = abs(metrics1['final_rmse'] - metrics2['final_rmse']) < 1e-10
    params_match = metrics1['final_parameters'] == metrics2['final_parameters']
    
    print(f"✅ Reproducibility test completed")
    print(f"   RMSE match: {'✅' if rmse_match else '❌'}")
    print(f"   Parameters match: {'✅' if params_match else '❌'}")
    print(f"   Run 1 RMSE: {metrics1['final_rmse']:.6f}")
    print(f"   Run 2 RMSE: {metrics2['final_rmse']:.6f}")
    print()

def example_parameter_analysis():
    """Analyze how parameters change during fitting."""
    print("🎬 Example 4: Parameter Analysis")
    print("-" * 30)
    
    metrics = create_fit_animation(
        seed=42,
        frames=100,
        fps=20,
        output_dir="examples/parameter_analysis",
        show_plot=False
    )
    
    params = metrics['final_parameters']
    print(f"✅ Parameter analysis completed")
    print(f"   L (maximum): {params['L']:.3f}")
    print(f"   k (growth rate): {params['k']:.3f}")
    print(f"   t0 (inflection): {params['t0']:.3f}")
    print(f"   t90 (90% completion): {params['t90']:.3f}")
    print(f"   Final RMSE: {metrics['final_rmse']:.4f}")
    print(f"   Correlation: {metrics['correlation']:.4f}")
    print()

def main():
    """Run all examples."""
    print("🚀 CapPredict Animation Module Examples")
    print("=" * 50)
    print()
    
    try:
        # Create examples directory
        os.makedirs("examples", exist_ok=True)
        
        # Run examples
        example_basic_usage()
        example_high_quality()
        example_reproducibility()
        example_parameter_analysis()
        
        print("🎉 All examples completed successfully!")
        print()
        print("📁 Check the 'examples' directory for generated animations")
        print("📊 Each example includes metrics.json with detailed statistics")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())