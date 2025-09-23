import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware


def main():
    """
    Poetry script entry point to run the FastAPI gateway server.
    """
    host = os.getenv("GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("GATEWAY_PORT", 8000))
    reload = os.getenv("GATEWAY_RELOAD", "true").lower() == "true"
    cors_origins = ["*"]

    app = uvicorn.create_asgi()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    print(f"Starting Gateway Server at http://{host}:{port} (Reload: {reload})")

    uvicorn.run(
        "llm_agent_x.server.gateway:app_asgi",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        app=app,
    )


if __name__ == "__main__":
    main()