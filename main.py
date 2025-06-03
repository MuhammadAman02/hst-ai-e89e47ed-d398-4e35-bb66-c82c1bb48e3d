"""
ML Engineer Portfolio - Main Application Entry Point
A professional, interactive portfolio for showcasing ML engineering skills and projects.
"""
import os
from dotenv import load_dotenv
from nicegui import ui

# Import the portfolio pages to register them with NiceGUI
import app.main  # noqa: F401

# Load environment variables from .env file (if present)
load_dotenv()

if __name__ in {"__main__", "__mp_main__"}:
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    
    ui.run(
        host=host,
        port=port,
        title="ML Engineer Portfolio",
        favicon="🧠",
        dark=os.getenv("DARK_MODE", "auto"),
        uvicorn_logging_level='info',
        reload=False  # Set to True during development, False for production
    )