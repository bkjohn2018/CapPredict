#!/usr/bin/env python3
"""
Animation module for demonstrating logistic S-curve fitting over time.

This module creates an animation showing how a logistic curve is fitted incrementally
as more data points are revealed, suitable for presentations and explanations.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

# Import our animation utilities
from .utils_animation import get_safe_animation_writer, get_writer_info

# Suppress scipy optimization warnings for cleaner output
warnings.filterwarnings("ignore", category=OptimizeWarning)

# Add CapPredict to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "CapPredict"))

try:
    from data_preprocessing import generate_curves
    from config import config
except ImportError:
    # Fallback if imports fail
    config = None


def logistic_function(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    """
    Logistic function for curve fitting.
    
    Args:
        t: Time points
        L: Maximum value (carrying capacity)
        k: Growth rate (steepness)
        t0: Inflection point (midpoint)
        
    Returns:
        Logistic curve values
    """
    return L / (1 + np.exp(-k * (t - t0)))


def generate_synthetic_scurve(
    time_points: int = 100,
    time_start: float = 0.0,
    time_end: float = 10.0,
    L: float = 1.0,
    k: float = 1.0,
    t0: float = 5.0,
    noise_std: float = 0.02,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic S-curve with optional noise.
    
    Args:
        time_points: Number of time points
        time_start: Start time
        time_end: End time
        L: Maximum value
        k: Growth rate
        t0: Inflection point
        noise_std: Standard deviation of noise to add
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (time_array, values_array)
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(time_start, time_end, time_points)
    y_clean = logistic_function(t, L, k, t0)
    
    # Add noise
    noise = np.random.normal(0, noise_std, len(t))
    y_noisy = y_clean + noise
    
    # Ensure values stay within reasonable bounds
    y_noisy = np.clip(y_noisy, 0, L * 1.1)
    
    return t, y_noisy


def fit_logistic_curve(
    t_data: np.ndarray, 
    y_data: np.ndarray
) -> Tuple[Optional[Tuple[float, float, float]], float]:
    """
    Fit a logistic curve to the given data.
    
    Args:
        t_data: Time points
        y_data: Observed values
        
    Returns:
        Tuple of (fitted_parameters, rmse) where parameters are (L, k, t0)
        Returns (None, inf) if fitting fails
    """
    if len(t_data) < 3:
        return None, float('inf')
    
    try:
        # Initial parameter guesses
        L_guess = max(y_data) * 1.1
        k_guess = 1.0
        t0_guess = np.mean(t_data)
        
        # Bounds for parameters (L > 0, k > 0, t0 in reasonable range)
        bounds = ([0.1, 0.01, t_data[0]], [L_guess * 2, 10.0, t_data[-1]])
        
        popt, _ = curve_fit(
            logistic_function,
            t_data,
            y_data,
            p0=[L_guess, k_guess, t0_guess],
            bounds=bounds,
            maxfev=1000
        )
        
        # Calculate RMSE
        y_pred = logistic_function(t_data, *popt)
        rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))
        
        return tuple(popt), rmse
        
    except Exception:
        return None, float('inf')


def calculate_t90(L: float, k: float, t0: float) -> float:
    """
    Calculate the time when the curve reaches 90% of its maximum value.
    
    Args:
        L: Maximum value
        k: Growth rate
        t0: Inflection point
        
    Returns:
        Time at 90% completion
    """
    # Solve: 0.9 * L = L / (1 + exp(-k * (t - t0)))
    # This gives: t = t0 + ln(9) / k
    return t0 + np.log(9) / k


class LogisticFitAnimator:
    """
    Creates an animation of incremental logistic curve fitting.
    """
    
    def __init__(
        self,
        time_data: np.ndarray,
        value_data: np.ndarray,
        frames: int = 150,
        fps: int = 24
    ):
        """
        Initialize the animator.
        
        Args:
            time_data: Full time series data
            value_data: Full value series data
            frames: Number of animation frames
            fps: Frames per second for animation
        """
        self.time_data = time_data
        self.value_data = value_data
        self.frames = frames
        self.fps = fps
        
        # Animation state
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.line_observed, = self.ax.plot([], [], 'bo', markersize=6, label='Observed Data')
        self.line_fitted, = self.ax.plot([], [], 'r-', linewidth=2, label='Fitted Logistic Curve')
        self.line_inflection = self.ax.axvline(x=0, color='orange', linestyle='--', alpha=0.7, label='Inflection Point (t₀)')
        self.line_t90 = self.ax.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='90% Completion (t₉₀)')
        
        # Text elements for parameter display
        self.text_params = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                       fontsize=12, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Setup plot
        self.setup_plot()
        
        # Store metrics for each frame
        self.metrics_history: List[Dict[str, Any]] = []
        
    def setup_plot(self):
        """Setup the plot appearance."""
        self.ax.set_xlim(self.time_data[0] - 0.5, self.time_data[-1] + 0.5)
        self.ax.set_ylim(-0.1, max(self.value_data) * 1.2)
        self.ax.set_xlabel('Time', fontsize=14)
        self.ax.set_ylabel('Progress', fontsize=14)
        self.ax.set_title('Incremental Logistic Curve Fitting', fontsize=16, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='center right', fontsize=10)
        
        # Style improvements
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
    def animate(self, frame: int) -> List:
        """
        Animation function called for each frame.
        
        Args:
            frame: Current frame number
            
        Returns:
            List of artists to update
        """
        # Calculate how many points to show (minimum 3 for fitting)
        n_points = max(3, int((frame + 1) / self.frames * len(self.time_data)))
        
        # Get current data subset
        t_current = self.time_data[:n_points]
        y_current = self.value_data[:n_points]
        
        # Update observed points
        self.line_observed.set_data(t_current, y_current)
        
        # Fit curve to current data
        params, rmse = fit_logistic_curve(t_current, y_current)
        
        if params is not None:
            L, k, t0 = params
            
            # Generate fitted curve over full domain
            t_full = np.linspace(self.time_data[0], self.time_data[-1], 200)
            y_fitted = logistic_function(t_full, L, k, t0)
            
            # Update fitted curve
            self.line_fitted.set_data(t_full, y_fitted)
            
            # Update inflection point line
            self.line_inflection.set_xdata([t0, t0])
            
            # Update t90 line
            t90 = calculate_t90(L, k, t0)
            self.line_t90.set_xdata([t90, t90])
            
            # Update parameter text
            param_text = (
                f'Points: {n_points}/{len(self.time_data)}\n'
                f'L (max): {L:.3f}\n'
                f'k (rate): {k:.3f}\n'
                f't₀ (inflection): {t0:.3f}\n'
                f't₉₀ (90%): {t90:.3f}\n'
                f'RMSE: {rmse:.4f}'
            )
            self.text_params.set_text(param_text)
            
            # Store metrics
            metrics = {
                'frame': frame,
                'n_points': n_points,
                'L': float(L),
                'k': float(k),
                't0': float(t0),
                't90': float(t90),
                'rmse': float(rmse)
            }
            self.metrics_history.append(metrics)
            
        else:
            # Fitting failed
            self.line_fitted.set_data([], [])
            self.line_inflection.set_xdata([0, 0])
            self.line_t90.set_xdata([0, 0])
            self.text_params.set_text(f'Points: {n_points}/{len(self.time_data)}\nFitting failed...')
            
            # Store failed metrics
            metrics = {
                'frame': frame,
                'n_points': n_points,
                'L': None,
                'k': None,
                't0': None,
                't90': None,
                'rmse': float('inf')
            }
            self.metrics_history.append(metrics)
        
        return [self.line_observed, self.line_fitted, self.line_inflection, 
                self.line_t90, self.text_params]
    
    def create_animation(self) -> animation.FuncAnimation:
        """
        Create the matplotlib animation object.
        
        Returns:
            FuncAnimation object
        """
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=self.frames,
            interval=1000//self.fps, blit=True, repeat=True
        )
        return anim
    
    def save_animation(self, output_path: str) -> str:
        """
        Save animation to file using the robust save_animation helper.
        
        Args:
            output_path: Path to save the animation file
            
        Returns:
            Path to the saved animation file
        """
        # Extract directory and basename from the output path
        output_path = Path(output_path)
        outdir = str(output_path.parent)
        basename = output_path.stem
        
        # Use the robust save_animation helper
        saved_path = save_animation(
            fig=self.fig,
            anim=self.create_animation(),
            outdir=outdir,
            basename=basename,
            fps=self.fps,
            dpi=120
        )
        
        return saved_path
    
    def save_metrics(self, metrics_path: str) -> None:
        """
        Save animation metrics to JSON file.
        
        Args:
            metrics_path: Path to save the metrics JSON file
        """
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")


def save_animation(fig, anim, outdir: str, basename: str, fps: int, dpi: int = 120) -> str:
    """
    Save animation with robust writer selection and timestamped directory.
    
    Args:
        fig: Matplotlib figure object
        anim: Animation object
        outdir: Base output directory
        basename: Base filename without extension
        fps: Frames per second
        dpi: Dots per inch for output
        
    Returns:
        Absolute path to the saved animation file
    """
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = Path(outdir) / f"{timestamp}_{basename}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get appropriate writer and extension
    writer, ext = get_safe_animation_writer(fps=fps, dpi=dpi)
    
    # Construct output path
    output_path = output_dir / f"{basename}{ext}"
    
    # Save animation
    print(f"[INFO] Writing animation to: {output_path}")
    print(f"[INFO] Using writer: {get_writer_info(writer)}")
    
    anim.save(str(output_path), writer=writer)
    
    # Close figure to free memory
    plt.close(fig)
    
    return str(output_path.absolute())


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Generate logistic curve fitting animation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--frames', type=int, default=150,
                       help='Number of animation frames')
    parser.add_argument('--fps', type=int, default=24,
                       help='Frames per second for animation')
    parser.add_argument('--time-points', type=int, default=100,
                       help='Number of time points in synthetic data')
    parser.add_argument('--noise', type=float, default=0.02,
                       help='Standard deviation of noise to add to synthetic data')
    
    args = parser.parse_args()
    
    print(f"🎬 Creating logistic curve fitting animation...")
    print(f"   Seed: {args.seed}")
    print(f"   Frames: {args.frames}")
    print(f"   FPS: {args.fps}")
    print(f"   Time points: {args.time_points}")
    
    # Generate synthetic S-curve data
    print("📈 Generating synthetic S-curve data...")
    
    # Try to use existing generator if available
    if config is not None:
        try:
            # Use existing generator with single curve
            curves_df = generate_curves(num_curves=1, seed=args.seed)
            time_data = curves_df[config.features.time_column].values
            value_data = curves_df.iloc[:, 0].values  # First curve
            print("   Using existing S-curve generator from CapPredict")
        except Exception as e:
            print(f"   Failed to use existing generator ({e}), falling back to synthetic")
            time_data, value_data = generate_synthetic_scurve(
                time_points=args.time_points,
                noise_std=args.noise,
                seed=args.seed
            )
    else:
        # Fallback to synthetic generator
        time_data, value_data = generate_synthetic_scurve(
            time_points=args.time_points,
            noise_std=args.noise,
            seed=args.seed
        )
        print("   Using fallback synthetic S-curve generator")
    
    # Create animator
    print("🎨 Creating animation...")
    animator = LogisticFitAnimator(time_data, value_data, args.frames, args.fps)
    
    # Save animation using the animator's save_animation method
    # Create artifacts directory in current working directory
    artifacts_dir = Path.cwd() / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    animation_path = animator.save_animation(str(artifacts_dir / "fit_demo"))
    
    # Save metrics in the same directory
    metrics_dir = Path(animation_path).parent
    metrics_path = metrics_dir / "metrics.json"
    animator.save_metrics(str(metrics_path))
    
    print(f"✅ Animation complete!")
    print(f"   Animation: {animation_path}")
    print(f"   Metrics: {metrics_path}")
    
    # Print the artifact path for CLI success
    print(f"\n🎯 Animation saved to: {animation_path}")
    
    # Return the animation path for CLI success
    return animation_path


if __name__ == "__main__":
    main()