"""OpenTelemetry tracing setup and utilities."""

import os
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from timestep.observability.exporters import (
    create_file_exporter,
    create_otlp_exporter,
    create_console_exporter,
)

_tracer_provider: Optional[TracerProvider] = None
_tracer: Optional[trace.Tracer] = None


def setup_tracing(
    exporter: str = "file",
    file_path: str = "traces.jsonl",
    otlp_endpoint: Optional[str] = None,
) -> None:
    """Setup OpenTelemetry tracing.
    
    Args:
        exporter: Exporter type ("file", "otlp", "console")
        file_path: Path for file exporter (default: traces.jsonl)
        otlp_endpoint: OTLP endpoint URL (required if exporter="otlp")
    """
    global _tracer_provider, _tracer
    
    # Create resource
    resource = Resource.create({
        "service.name": "timestep",
    })
    
    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)
    
    # Create exporter
    if exporter == "file":
        span_exporter = create_file_exporter(file_path)
    elif exporter == "otlp":
        if not otlp_endpoint:
            raise ValueError("otlp_endpoint is required when exporter='otlp'")
        span_exporter = create_otlp_exporter(otlp_endpoint)
    elif exporter == "console":
        span_exporter = create_console_exporter()
    else:
        raise ValueError(f"Unknown exporter: {exporter}")
    
    # Add span processor
    _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    
    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    
    # Create tracer
    _tracer = trace.get_tracer(__name__)


def get_tracer(name: str = __name__) -> trace.Tracer:
    """Get a tracer instance.
    
    Args:
        name: Tracer name
        
    Returns:
        Tracer instance
    """
    if _tracer is None:
        # Setup default tracing if not already done
        setup_tracing()
    
    return trace.get_tracer(name)
