"""
Backend entry point for Vercel serverless deployment.

Imports and re-exports the FastAPI app handler.
"""

from app.main import handler, app, health

# Export the Mangum handler for Vercel
__all__ = ["handler", "app", "health"]