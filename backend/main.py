"""
Backend entry point for Render deployment.
Imports the FastAPI app from app.main module.
"""

from app.main import app

# This allows uvicorn main:app to work when Root Directory is set to 'backend'
__all__ = ["app"]