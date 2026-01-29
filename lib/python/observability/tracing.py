"""OpenTelemetry tracing helpers for observability."""

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

logger = logging.getLogger(__name__)

# Configure logger to suppress noisy OTLP export errors
export_logger = logging.getLogger("opentelemetry.sdk.trace.export")
export_logger.setLevel(logging.WARNING)


def configure_tracing() -> None:
    """Configure OpenTelemetry tracing for OpenAI calls.
    
    Configuration is read from standard OTEL environment variables:
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    - OTEL_EXPORTER_OTLP_HEADERS: Headers for OTLP export (optional)
    - OTEL_RESOURCE_ATTRIBUTES: Resource attributes (optional)
    """
    # Check if already initialized
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        logger.debug("Tracing already initialized, skipping")
        return
    
    # Create SDK components - they auto-read from env vars
    resource = Resource.create({})  # Auto-reads from OTEL_RESOURCE_ATTRIBUTES
    exporter = OTLPSpanExporter()  # Auto-reads from OTEL_EXPORTER_OTLP_ENDPOINT
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)
    
    # Instrument OpenAI
    OpenAIInstrumentor().instrument()
    logger.info("OpenTelemetry tracing configured for OpenAI calls")
