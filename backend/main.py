"""
Backend entry point for Vercel serverless deployment.

Imports and re-exports the FastAPI app handler.
Sets up Python path for proper module resolution.
"""

import sys
import os

# Add the backend directory to Python path for Vercel deployment
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Now import from the app module
from app.main import handler, app, health  # noqa: E402

# Export the Mangum handler for Vercel
__all__ = ["handler", "app", "health"]