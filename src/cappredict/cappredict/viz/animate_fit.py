"""
Logistic Curve Fitting Animation Module

This module creates animations demonstrating how a logistic S-curve is fitted
to data points as they are progressively revealed over time.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def logistic_function(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    """
    Logistic function for S-curve modeling.
    
    Args:
        t: Time points
        L: Maximum value (carrying capacity)
        k: Growth rate (steepness)
        t0: Inflection point (time of maximum growth rate)
    
    Returns:
        Array of logistic function values
    """
    return L / (1 + np.exp(-k * (t - t0)))


def generate_synthetic_curve(
    time_points: int = 100,
    time_range: Tuple[float, float] = (0, 10),
    L: float = 1.0,
    k: float = 1.5,
    t0: float = 5.0,
    noise_level: float = 0.02,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic logistic curve with optional noise.
    
    Args:
        time_points: Number of time points
        time_range: Tuple of (start_time, end_time)
        L: Maximum value
        k: Growth rate
        t0: Inflection point
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (time_array, values_array)
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(time_range[0], time_range[1], time_points)
    y_true = logistic_function(t, L, k, t0)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, time_points)
    y_noisy = np.clip(y_true + noise, 0, L)  # Ensure values stay in [0, L]
    
    return t, y_noisy


def fit_logistic_curve(
    t: np.ndarray, 
    y: np.ndarray, 
    initial_guess: Tuple[float, float, float] = (1.0, 1.0, 5.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a logistic curve to the given data points.
    
    Args:
        t: Time points
        y: Observed values
        initial_guess: Initial parameter guess (L, k, t0)
    
    Returns:
        Tuple of (fitted_parameters, parameter_covariance)
    """
    try:
        # Set bounds to ensure physically meaningful parameters
        bounds = (
            [0.1, 0.1, -np.inf],  # Lower bounds: L > 0, k > 0, t0 can be any
            [np.inf, np.inf, np.inf]  # Upper bounds: no upper limit
        )
        
        popt, pcov = curve_fit(
            logistic_function, 
            t, 
            y, 
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        return popt, pcov
    except (RuntimeError, ValueError):
        # Return initial guess if fitting fails
        return np.array(initial_guess), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error between true and predicted values."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_t90(L: float, k: float, t0: float) -> float:
    """Calculate time to reach 90% of maximum value."""
    return t0 + np.log(9) / k


def create_fit_animation(
    seed: int = 42,
    frames: int = 150,
    fps: int = 24,
    output_dir: str = "artifacts",
    show_plot: bool = False
) -> Dict[str, Any]:
    """
    Create an animation showing progressive logistic curve fitting.
    
    Args:
        seed: Random seed for reproducibility
        frames: Number of animation frames
        fps: Frames per second for the animation
        output_dir: Directory to save output files
        show_plot: Whether to display the plot interactively
    
    Returns:
        Dictionary containing metrics and file paths
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate synthetic data
    t_full, y_full = generate_synthetic_curve(
        time_points=100,
        time_range=(0, 10),
        L=1.0,
        k=1.2,
        t0=5.0,
        noise_level=0.02,
        seed=seed
    )
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.style.use('default')  # Clean, professional style
    
    # Set up the plot
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Progress', fontsize=12, fontweight='bold')
    ax.set_title('Logistic Curve Fitting Demonstration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Initialize plot elements
    observed_line, = ax.plot([], [], 'o', color='#1f77b4', markersize=6, label='Observed Data')
    fitted_line, = ax.plot([], [], '-', color='#ff7f0e', linewidth=2, label='Fitted Curve')
    inflection_line = ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Inflection Point (t0)')
    t90_line = ax.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='90% Completion (t90)')
    
    # Text box for parameters
    text_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    def animate(frame):
        """Animation function for each frame."""
        # Calculate how many points to show (progressive reveal)
        num_points = max(3, int((frame + 1) * len(t_full) / frames))
        t_visible = t_full[:num_points]
        y_visible = y_full[:num_points]
        
        # Update observed data
        observed_line.set_data(t_visible, y_visible)
        
        # Fit curve to visible data
        if len(t_visible) >= 3:
            try:
                popt, pcov = fit_logistic_curve(t_visible, y_visible)
                L_fit, k_fit, t0_fit = popt
                
                # Generate fitted curve across full domain
                t_fit = np.linspace(0, 10, 200)
                y_fit = logistic_function(t_fit, L_fit, k_fit, t0_fit)
                
                # Update fitted curve
                fitted_line.set_data(t_fit, y_fit)
                
                # Update vertical lines
                inflection_line.set_xdata(t0_fit)
                t90_fit = calculate_t90(L_fit, k_fit, t0_fit)
                t90_line.set_xdata(t90_fit)
                
                # Calculate RMSE for visible points
                y_pred_visible = logistic_function(t_visible, L_fit, k_fit, t0_fit)
                rmse = calculate_rmse(y_visible, y_pred_visible)
                
                # Update text box
                text_content = f'Parameters:\nL (max): {L_fit:.3f}\nk (rate): {k_fit:.3f}\nt0 (inflection): {t0_fit:.3f}\n\nRMSE: {rmse:.4f}\nPoints: {num_points}'
                text_box.set_text(text_content)
                
            except Exception:
                # If fitting fails, keep previous state
                pass
        
        return observed_line, fitted_line, inflection_line, t90_line, text_box
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=1000/fps, 
        blit=False, repeat=False
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    
    # Save animation as MP4
    mp4_path = os.path.join(output_path, "fit_demo.mp4")
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='CapPredict'))
    anim.save(mp4_path, writer=writer, dpi=100)
    
    # Calculate final metrics
    final_popt, final_pcov = fit_logistic_curve(t_full, y_full)
    L_final, k_final, t0_final = final_popt
    y_pred_final = logistic_function(t_full, L_final, k_final, t0_final)
    final_rmse = calculate_rmse(y_full, y_pred_final)
    final_t90 = calculate_t90(L_final, k_final, t0_final)
    
    # Calculate correlation
    correlation, _ = pearsonr(y_full, y_pred_final)
    
    # Prepare metrics
    metrics = {
        "seed": seed,
        "frames": frames,
        "fps": fps,
        "timestamp": timestamp,
        "final_parameters": {
            "L": float(L_final),
            "k": float(k_final),
            "t0": float(t0_final),
            "t90": float(final_t90)
        },
        "final_rmse": float(final_rmse),
        "correlation": float(correlation),
        "data_points": len(t_full),
        "output_files": {
            "mp4": mp4_path,
            "metrics": os.path.join(output_path, "metrics.json")
        }
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_path, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    return metrics


def main():
    """Command line interface for the animation module."""
    parser = argparse.ArgumentParser(
        description="Create logistic curve fitting animation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--frames", 
        type=int, 
        default=150,
        help="Number of animation frames"
    )
    
    parser.add_argument(
        "--fps", 
        type=int, 
        default=24,
        help="Frames per second for the animation"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="artifacts",
        help="Output directory for generated files"
    )
    
    parser.add_argument(
        "--show-plot", 
        action="store_true",
        help="Display the plot interactively"
    )
    
    args = parser.parse_args()
    
    print(f"🎬 Creating logistic curve fitting animation...")
    print(f"   Seed: {args.seed}")
    print(f"   Frames: {args.frames}")
    print(f"   FPS: {args.fps}")
    print(f"   Output directory: {args.output_dir}")
    
    try:
        metrics = create_fit_animation(
            seed=args.seed,
            frames=args.frames,
            fps=args.fps,
            output_dir=args.output_dir,
            show_plot=args.show_plot
        )
        
        print(f"✅ Animation created successfully!")
        print(f"   MP4 saved to: {metrics['output_files']['mp4']}")
        print(f"   Metrics saved to: {metrics['output_files']['metrics']}")
        print(f"   Final RMSE: {metrics['final_rmse']:.4f}")
        print(f"   Correlation: {metrics['correlation']:.4f}")
        
    except Exception as e:
        print(f"❌ Error creating animation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())