#!/usr/bin/env python3
"""
Simple test runner for CapPredict package.

Run with: python run_tests.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        import pytest
        
        # Run tests
        print("🧪 Running CapPredict tests...")
        result = pytest.main([
            "tests/",
            "-v",
            "--tb=short"
        ])
        
        if result == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
            
    except ImportError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)