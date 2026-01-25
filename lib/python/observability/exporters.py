"""OpenTelemetry exporters configuration."""

import json
from pathlib import Path
from typing import Optional
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.resources import Resource


class FileSpanExporter(SpanExporter):
    """Custom file exporter that writes spans to JSONL format."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(self.file_path, "a")
    
    def export(self, spans):
        """Export spans to file in JSONL format."""
        for span in spans:
            # Convert span to dict format
            span_dict = {
                "name": span.name,
                "context": {
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                },
                "kind": str(span.kind),
                "start_time": span.start_time,
                "end_time": span.end_time,
                "attributes": dict(span.attributes) if span.attributes else {},
                "events": [
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": dict(event.attributes) if event.attributes else {},
                    }
                    for event in (span.events or [])
                ],
                "status": {
                    "status_code": span.status.status_code.name if span.status else "UNSET",
                    "description": span.status.description if span.status else None,
                },
            }
            
            # Write as JSONL (one JSON object per line)
            json.dump(span_dict, self.file_handle)
            self.file_handle.write("\n")
            self.file_handle.flush()
        
        return SpanExportResult.SUCCESS
    
    def shutdown(self):
        """Shutdown the exporter."""
        if self.file_handle:
            self.file_handle.close()


def create_file_exporter(file_path: str = "traces.jsonl"):
    """Create a file exporter that writes traces to JSONL format.
    
    Args:
        file_path: Path to write traces to (default: traces.jsonl)
        
    Returns:
        FileSpanExporter configured for file output
    """
    return FileSpanExporter(file_path)


def create_otlp_exporter(endpoint: str = "http://localhost:4318/v1/traces"):
    """Create an OTLP exporter for sending traces to an OTLP endpoint.
    
    Args:
        endpoint: OTLP endpoint URL
        
    Returns:
        OTLPSpanExporter configured for the endpoint
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(endpoint=endpoint)
    except ImportError:
        raise ImportError(
            "OTLP exporter requires opentelemetry-exporter-otlp-proto-http. "
            "Install it with: pip install opentelemetry-exporter-otlp-proto-http"
        )


def create_console_exporter():
    """Create a console exporter for printing traces to stdout.
    
    Returns:
        ConsoleSpanExporter
    """
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    return ConsoleSpanExporter()
