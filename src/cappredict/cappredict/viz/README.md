# CapPredict Visualization Module

This module provides visualization tools for the CapPredict machine learning pipeline, including an interactive animation that demonstrates logistic curve fitting.

## Features

### Logistic Curve Fitting Animation

The `animate_fit.py` module creates professional animations showing how a logistic S-curve is progressively fitted to data points as they are revealed over time. This is perfect for:

- **Presentations**: Demonstrate the fitting process to colleagues
- **Education**: Show how machine learning models learn from data
- **Documentation**: Visualize the capacity prediction methodology

## Usage

### Command Line Interface

The easiest way to use the animation module is through the command line:

```bash
# Basic usage with default parameters
python -m cappredict.viz.animate_fit

# Custom parameters
python -m cappredict.viz.animate_fit --seed 42 --frames 150 --fps 24

# Show the plot interactively
python -m cappredict.viz.animate_fit --show-plot
```

### Command Line Options

- `--seed`: Random seed for reproducibility (default: 42)
- `--frames`: Number of animation frames (default: 150)
- `--fps`: Frames per second for the animation (default: 24)
- `--output-dir`: Output directory for generated files (default: "artifacts")
- `--show-plot`: Display the plot interactively

### Programmatic Usage

```python
from cappredict.viz.animate_fit import create_fit_animation

# Create animation
metrics = create_fit_animation(
    seed=42,
    frames=150,
    fps=24,
    output_dir="artifacts",
    show_plot=False
)

# Access results
print(f"Final RMSE: {metrics['final_rmse']:.4f}")
print(f"MP4 saved to: {metrics['output_files']['mp4']}")
```

## Output

The module generates:

1. **MP4 Animation**: High-quality video showing the fitting process
2. **Metrics JSON**: Detailed statistics and parameters
3. **Timestamped Directory**: Organized output structure

### Output Structure

```
artifacts/
└── 20241201_143022/
    ├── fit_demo.mp4          # Animation video
    └── metrics.json          # Detailed metrics
```

### Metrics Included

- **Parameters**: L (maximum), k (growth rate), t0 (inflection point), t90 (90% completion)
- **Quality Metrics**: RMSE, correlation coefficient
- **Animation Settings**: Seed, frames, FPS
- **File Paths**: Locations of generated files

## Animation Features

### Visual Elements

- **Observed Data**: Blue dots showing progressively revealed data points
- **Fitted Curve**: Orange line showing current best-fit logistic function
- **Inflection Point**: Red dashed line marking the steepest growth point
- **90% Completion**: Green dashed line showing time to near-completion
- **Live Parameters**: Real-time display of fitted parameters and RMSE

### Technical Details

- **Progressive Reveal**: Data points are shown incrementally
- **Real-time Fitting**: Logistic curve is refitted at each frame
- **Professional Styling**: Clean, presentation-ready visuals
- **Deterministic**: Same seed always produces identical results

## Requirements

- Python 3.8+
- matplotlib
- scipy
- numpy
- FFmpeg (for MP4 export)

## Installation

```bash
# Install the package
pip install -e .

# Or install dependencies manually
pip install matplotlib scipy numpy
```

## Examples

### Quick Demo

```bash
# Run a short demo animation
python demo_animation.py
```

### Custom Animation

```python
from cappredict.viz.animate_fit import create_fit_animation

# Create a high-quality animation for presentation
metrics = create_fit_animation(
    seed=123,
    frames=300,    # Smooth animation
    fps=30,        # Professional frame rate
    output_dir="presentation_assets"
)
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python run_tests.py

# Run specific test file
pytest tests/test_animate_fit.py -v
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg for MP4 export
2. **Memory issues**: Reduce frame count for large animations
3. **Display issues**: Use `--show-plot` for interactive viewing

### Performance Tips

- Use fewer frames for quick previews
- Lower FPS for smaller file sizes
- Close other applications during long animations

## Contributing

When adding new visualization features:

1. Follow the existing code style
2. Add comprehensive tests
3. Update this README
4. Include type hints and docstrings

## License

This module is part of the CapPredict package and follows the same license terms.