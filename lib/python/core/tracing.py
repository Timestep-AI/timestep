"""OpenTelemetry tracing configuration module.

This module provides zero-code instrumentation setup for OpenTelemetry tracing.
It can be enabled/disabled via environment variables and requires minimal code changes.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry components
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.debug("OpenTelemetry not available, tracing will be disabled")


def setup_tracing(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Initialize OpenTelemetry tracing with OTLP exporter for Jaeger/Tempo.
    
    This function sets up OpenTelemetry tracing with programmatic instrumentation
    and an OTLP exporter. It can be configured via environment variables
    or function parameters.
    
    Args:
        service_name: Service name for traces (defaults to OTEL_SERVICE_NAME env var or "timestep")
        otlp_endpoint: OTLP endpoint URL (defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var or "http://localhost:4317")
        enabled: Whether to enable tracing (defaults to OTEL_ENABLED env var or True)
        
    Returns:
        True if tracing was successfully initialized, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.debug("OpenTelemetry not available, skipping tracing setup")
        return False
    
    # Disable OpenTelemetry auto-instrumentation to prevent conflicts
    # We use programmatic instrumentation instead
    # This prevents opentelemetry-distro from auto-initializing
    if "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS" not in os.environ:
        os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = "all"
    
    # Check if tracing is enabled
    if enabled is None:
        enabled = os.getenv("OTEL_ENABLED", "true").lower() in ("true", "1", "yes")
    
    if not enabled:
        logger.debug("Tracing is disabled via configuration")
        return False
    
    # Check if already initialized (avoid double initialization)
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        logger.debug("Tracing already initialized, skipping")
        return True
    
    try:
        # Get configuration from environment or parameters
        if service_name is None:
            service_name = os.getenv("OTEL_SERVICE_NAME", "timestep")
        
        if otlp_endpoint is None:
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        
        # Create resource with service name
        resource = Resource.create({
            SERVICE_NAME: service_name,
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Create OTLP exporter
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        except ImportError:
            logger.warning("OTLP gRPC exporter not available, trying HTTP exporter")
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            except ImportError:
                logger.warning("OTLP HTTP exporter not available, using console exporter")
                try:
                    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                    otlp_exporter = ConsoleSpanExporter()
                    logger.warning("Falling back to console exporter. Install opentelemetry-exporter-otlp-proto-grpc for Jaeger support.")
                except ImportError:
                    logger.error("No span exporter available, tracing will not work")
                    return False
        
        # Add span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        logger.info(
            f"OpenTelemetry tracing initialized: service={service_name}, "
            f"otlp_endpoint={otlp_endpoint}"
        )
        
        # Try to enable programmatic instrumentation for common libraries
        _setup_programmatic_instrumentation()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}", exc_info=True)
        return False


def _setup_programmatic_instrumentation():
    """Set up programmatic instrumentation for common libraries.
    
    This function attempts to enable programmatic instrumentation for:
    - HTTP clients (httpx, requests)
    - OpenAI (if available)
    
    Note: FastAPI instrumentation is done separately via instrument_fastapi_app()
    to ensure the app is created first.
    """
    if not OPENTELEMETRY_AVAILABLE:
        return
    
    try:
        # HTTP client instrumentation
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument()
            logger.debug("HTTPX client instrumentation enabled")
        except ImportError:
            logger.debug("HTTPX client instrumentor not available")
        
        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            RequestsInstrumentor().instrument()
            logger.debug("Requests client instrumentation enabled")
        except ImportError:
            logger.debug("Requests client instrumentor not available")
        
        # OpenAI instrumentation (if available)
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
            logger.debug("OpenAI client instrumentation enabled")
        except ImportError:
            logger.debug("OpenAI client instrumentor not available")
            
    except Exception as e:
        logger.warning(f"Error setting up programmatic instrumentation: {e}", exc_info=True)


def instrument_fastapi_app(app):
    """Instrument a FastAPI application for tracing.
    
    This should be called after creating the FastAPI app instance.
    
    Args:
        app: FastAPI application instance
    """
    if not OPENTELEMETRY_AVAILABLE:
        return
    
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.debug("FastAPI app instrumented for tracing")
    except ImportError:
        logger.debug("FastAPI instrumentor not available")
    except Exception as e:
        logger.warning(f"Error instrumenting FastAPI app: {e}", exc_info=True)
