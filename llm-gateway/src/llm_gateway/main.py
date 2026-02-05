"""FastAPI application entry point.

Creates and configures the FastAPI application with all
middleware, routes, and lifecycle events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_gateway import __version__
from llm_gateway.config import get_gateway_settings
from llm_gateway.gateway import LLMGateway
from llm_gateway.api.routes import router
from llm_gateway.api.middleware import setup_middleware
from llm_gateway.resilience.rate_limiter import TokenBucketRateLimiter


# Configure logging
def configure_logging() -> None:
    """Configure application logging."""
    settings = get_gateway_settings()

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        if settings.log_format == "text"
        else '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler()],
    )

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    settings = get_gateway_settings()

    # Initialize rate limiter
    rate_limiter = None
    if settings.rate_limit_enabled:
        rate_limiter = TokenBucketRateLimiter(
            requests_per_minute=settings.rate_limit_requests_per_minute,
        )

    # Initialize gateway
    gateway = LLMGateway(
        settings=settings,
        rate_limiter=rate_limiter,
    )

    # Store in app state
    app.state.gateway = gateway
    app.state.rate_limiter = rate_limiter
    app.state.settings = settings

    # Startup
    logger.info(f"Starting LLM Gateway v{__version__}")
    await gateway.startup()

    yield

    # Shutdown
    logger.info("Shutting down LLM Gateway")
    await gateway.shutdown()


def create_app() -> FastAPI:
    """Create the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    settings = get_gateway_settings()

    app = FastAPI(
        title="LLM Gateway",
        description="Production-grade LLM Gateway with multi-provider routing, "
        "intelligent failover, rate limiting, and semantic caching.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    setup_middleware(app)

    # Include routes
    app.include_router(router)

    return app


# Create application instance
app = create_app()


def main() -> None:
    """Run the application using uvicorn."""
    import uvicorn

    settings = get_gateway_settings()

    uvicorn.run(
        "llm_gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
