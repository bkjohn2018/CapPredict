#!/usr/bin/env python3
"""
Setup script for CapPredict package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cappredict",
    version="0.1.0",
    author="CapPredict Team",
    description="Capacity Prediction using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "imbalanced-learn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pylint>=2.17.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cappredict-animate=cappredict.viz.animate_fit:main",
        ],
    },
)