"""IV Surface - Interactive volatility surface visualization."""

from iv_surface.app import app


def main() -> None:
    """Entry point for the application."""
    import uvicorn

    uvicorn.run(
        "iv_surface.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


__all__ = ["app", "main"]
