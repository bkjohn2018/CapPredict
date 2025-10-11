#!/usr/bin/env python3
"""
Animation utility functions for CapPredict visualization modules.

This module provides robust animation writer selection with automatic fallback
from FFmpeg (MP4) to Pillow (GIF) when FFmpeg is unavailable.
"""

import logging
import shutil
from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_safe_animation_writer(fps: int = 24, dpi: int = 120) -> Tuple[animation.AbstractMovieWriter, str]:
    """
    Get a safe animation writer with automatic FFmpeg fallback.
    
    Args:
        fps: Frames per second for the animation
        dpi: Dots per inch for the output
        
    Returns:
        Tuple of (writer, file_extension) where writer is a matplotlib animation writer
        and file_extension is the appropriate file extension (".mp4" or ".gif")
    """
    # Try to use FFmpeg for high-quality MP4 output
    if shutil.which("ffmpeg") is not None:
        try:
            # Set matplotlib's FFmpeg path
            plt.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg")
            
            # Create FFmpeg writer with optimal settings for compatibility
            writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=[
                    "-pix_fmt", "yuv420p",  # Ensures compatibility with most players
                    "-movflags", "+faststart",  # Optimizes streaming
                    "-preset", "medium"  # Good balance of speed vs quality
                ],
                bitrate=1800
            )
            
            logger.info("Using FFmpeg writer (mp4, libx264, yuv420p)")
            return writer, ".mp4"
            
        except Exception as e:
            logger.warning(f"FFmpeg writer failed: {e}. Falling back to GIF.")
    
    # Fallback to GIF using Pillow
    logger.info("Falling back to GIF writer (no FFmpeg).")
    writer = animation.PillowWriter(fps=fps)
    return writer, ".gif"


def get_writer_info(writer: animation.AbstractMovieWriter) -> str:
    """
    Get human-readable information about the animation writer.
    
    Args:
        writer: The matplotlib animation writer
        
    Returns:
        String describing the writer type and capabilities
    """
    if isinstance(writer, animation.FFMpegWriter):
        return f"FFmpeg MP4 (libx264, yuv420p, {writer.fps} fps)"
    elif isinstance(writer, animation.PillowWriter):
        return f"Pillow GIF ({writer.fps} fps)"
    else:
        return f"Unknown writer: {type(writer).__name__}"
