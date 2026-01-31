"""OpenTelemetry tracing helpers for observability."""

import logging
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

# Configure logger to suppress noisy OTLP export errors
export_logger = logging.getLogger("opentelemetry.sdk.trace.export")
export_logger.setLevel(logging.WARNING)

# Optional instrumentations - gracefully handle if not installed
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except ImportError:
    FastAPIInstrumentor = None

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    logger.debug("httpx instrumentation module imported successfully")
except ImportError as e:
    HTTPXClientInstrumentor = None
    logger.warning(f"Failed to import httpx instrumentation: {e}")

# Ensure W3C Trace Context propagator is configured
# This is critical for distributed tracing across HTTP boundaries
try:
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositeHTTPPropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    
    # Set up W3C Trace Context propagator (default, but ensure it's set)
    propagators = os.getenv("OTEL_PROPAGATORS", "tracecontext,baggage").split(",")
    if "tracecontext" in propagators or not propagators:
        # Ensure tracecontext propagator is set (this is the default)
        # The SDK should handle this automatically, but we ensure it's configured
        pass  # SDK handles this via OTEL_PROPAGATORS env var
except ImportError:
    # If propagator modules aren't available, SDK will use defaults
    pass


def configure_tracing(app=None) -> None:
    """Configure OpenTelemetry tracing for OpenAI calls, FastAPI, and HTTP clients.
    
    Configuration is read from standard OTEL environment variables:
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    - OTEL_EXPORTER_OTLP_HEADERS: Headers for OTLP export (optional)
    - OTEL_RESOURCE_ATTRIBUTES: Resource attributes (optional)
    - OTEL_PROPAGATORS: Propagators to use (default: tracecontext,baggage)
    - OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: Set to 'true' to capture message content (handled by instrumentation)
    
    Args:
        app: Optional FastAPI app instance to instrument. If provided, instruments this specific app
             with excluded internal routes (/mcp, /.well-known). If None, does not instrument FastAPI.
    """
    # Check if already initialized
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        logger.debug("Tracing already initialized")
        # If app is provided and FastAPI instrumentation is available, instrument it
        if app is not None and FastAPIInstrumentor is not None:
            try:
                excluded_urls = "/mcp,/.well-known"
                FastAPIInstrumentor().instrument_app(app, excluded_urls=excluded_urls)
                logger.info(f"FastAPI app instrumented (excluding: {excluded_urls})")
            except Exception as e:
                logger.warning(f"Failed to instrument FastAPI app: {e}")
        return
    
    # Create SDK components - they auto-read from env vars
    resource = Resource.create({})  # Auto-reads from OTEL_RESOURCE_ATTRIBUTES
    exporter = OTLPSpanExporter()  # Auto-reads from OTEL_EXPORTER_OTLP_ENDPOINT
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)
    
    # Instrument OpenAI
    # The instrumentation automatically reads OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT from environment
    try:
        from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
        logger.info("OpenAI instrumentation enabled")
    except Exception as e:
        logger.error(f"Failed to instrument OpenAI: {e}", exc_info=True)
    
    # Instrument httpx to propagate trace context in outgoing HTTP requests
    # CRITICAL: This must be called before any httpx clients are created
    if HTTPXClientInstrumentor is not None:
        try:
            HTTPXClientInstrumentor().instrument()
            logger.info("httpx instrumentation enabled - trace context will propagate in HTTP requests")
        except Exception as e:
            logger.error(f"Failed to instrument httpx: {e}", exc_info=True)
    else:
        logger.warning("httpx instrumentation not available - install opentelemetry-instrumentation-httpx")
        logger.warning("Without httpx instrumentation, HTTP requests will not propagate trace context")
    
    # Instrument FastAPI app if provided, excluding internal routes
    if app is not None and FastAPIInstrumentor is not None:
        try:
            # Exclude internal routes to avoid creating separate traces for agent-to-environment communication
            # excluded_urls must be a comma-separated string, not a list
            excluded_urls = "/mcp,/.well-known"
            FastAPIInstrumentor().instrument_app(app, excluded_urls=excluded_urls)
            logger.info(f"FastAPI app instrumented (excluding: {excluded_urls})")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI app: {e}")
    
    logger.info("OpenTelemetry tracing configured")
