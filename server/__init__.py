# server package — exposes the FastAPI app for uvicorn and openenv runner
from server.app import app, main

__all__ = ["app", "main"]