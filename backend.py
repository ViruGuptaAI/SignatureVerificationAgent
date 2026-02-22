"""
Local development entry point.

  Local:  python backend.py
  Azure:  Configured via startup command (see startup.txt)
"""

from app.main import app  
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info", workers=4)