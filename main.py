"""
AR Face Filter Application - Entry Point

A real-time augmented reality face filter application that overlays
PNG assets onto facial landmarks detected via webcam.

Usage:
    python main.py

Controls:
    - Press 'q' to quit

Requirements:
    - Webcam connected
    - PNG filter assets in the 'assets' directory
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.filter_app import FilterApplication


def main() -> None:
    """Entry point for the AR Face Filter application."""
    print("=" * 60)
    print("AR Face Filter Application")
    print("=" * 60)
    print()

    # Create and run application
    app = FilterApplication()

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.cleanup()

    print("\nApplication closed.")


if __name__ == "__main__":
    main()
