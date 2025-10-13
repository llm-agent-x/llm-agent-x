import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


def main():
    """
    Poetry script entry point to run the FastAPI gateway server.
    """
    host = os.getenv("GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("GATEWAY_PORT", 8000))
    reload = os.getenv("GATEWAY_RELOAD", "true").lower() == "true"

    print(f"Starting Gateway Server at http://{host}:{port} (Reload: {reload})")

    uvicorn.run(
        "llm_agent_x.server.gateway:app_asgi",
        host=host,
        port=port,
        reload=reload,
        log_level="debug",
        # app=app,
    )


if __name__ == "__main__":
    main()
