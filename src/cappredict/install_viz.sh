#!/bin/bash
# Installation script for CapPredict visualization module

echo "🚀 Installing CapPredict visualization module dependencies..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $python_version"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements_viz.txt

# Check if FFmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. MP4 export will not work."
    echo "   Install FFmpeg for MP4 export functionality:"
    echo "   - Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "   - CentOS/RHEL: sudo yum install ffmpeg"
    echo "   - macOS: brew install ffmpeg"
    echo "   - Windows: Download from https://ffmpeg.org/"
else
    echo "✅ FFmpeg found: $(ffmpeg -version | head -n1)"
fi

# Install the package in development mode
echo "🔧 Installing CapPredict package in development mode..."
pip3 install -e .

echo "✅ Installation complete!"
echo ""
echo "🎬 To test the animation module:"
echo "   python3 demo_animation.py"
echo ""
echo "🧪 To run tests:"
echo "   python3 run_tests.py"
echo ""
echo "📖 For more information, see: cappredict/viz/README.md"