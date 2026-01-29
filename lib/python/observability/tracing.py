"""OpenTelemetry tracing helpers for observability."""

import logging
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
    OpenAIInstrumentor().instrument()
    logger.info("OpenTelemetry tracing configured for OpenAI calls")
