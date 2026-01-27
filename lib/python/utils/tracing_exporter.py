"""Custom OpenTelemetry file exporter for writing traces to JSON Lines format."""

import json
import logging
import os
from pathlib import Path
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)

try:
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.trace import Span
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Graceful degradation if OpenTelemetry not installed
    SpanExporter = object
    SpanExportResult = type('SpanExportResult', (), {
        'SUCCESS': 'SUCCESS',
        'FAILURE': 'FAILURE'
    })()
    Span = None
    OPENTELEMETRY_AVAILABLE = False


class FileSpanExporter(SpanExporter):
    """OpenTelemetry span exporter that writes traces to a file in JSON Lines format.
    
    Each line contains a JSON object representing a complete trace with all its spans.
    This exporter collects spans until a trace is complete, then writes it to file.
    """
    
    def __init__(self, file_path: str = "traces.jsonl"):
        if not OPENTELEMETRY_AVAILABLE:
            raise ImportError("OpenTelemetry is not available")
        """Initialize the file exporter.
        
        Args:
            file_path: Path to the output file (default: "traces.jsonl")
        """
        self.file_path = Path(file_path)
        self.file_lock = Lock()
        
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file in append mode
        self._file = open(self.file_path, "a", encoding="utf-8")
        
        logger.info(f"FileSpanExporter initialized, writing to {self.file_path}")
    
    def export(self, spans):
        """Export spans to file.
        
        Args:
            spans: List of spans to export
            
        Returns:
            SpanExportResult.SUCCESS
        """
        if not spans:
            return SpanExportResult.SUCCESS if OPENTELEMETRY_AVAILABLE else "SUCCESS"
        
        try:
            with self.file_lock:
                # Group spans by trace_id
                traces = {}
                for span in spans:
                    trace_id = format(span.context.trace_id, "032x")
                    if trace_id not in traces:
                        traces[trace_id] = {
                            "trace_id": trace_id,
                            "spans": [],
                            "resource": {}
                        }
                    
                    # Convert span to dict
                    span_dict = self._span_to_dict(span)
                    traces[trace_id]["spans"].append(span_dict)
                
                # Write each trace as a JSON line
                for trace in traces.values():
                    # Sort spans by start time
                    trace["spans"].sort(key=lambda s: s.get("start_time", 0))
                    json_line = json.dumps(trace, default=str)
                    self._file.write(json_line + "\n")
                    self._file.flush()
                
                return SpanExportResult.SUCCESS if OPENTELEMETRY_AVAILABLE else "SUCCESS"
        except Exception as e:
            logger.error(f"Error exporting spans to file: {e}", exc_info=True)
            return SpanExportResult.FAILURE if OPENTELEMETRY_AVAILABLE else "FAILURE"
    
    def _span_to_dict(self, span: Span) -> dict:
        """Convert a span to a dictionary representation.
        
        Args:
            span: OpenTelemetry span object
            
        Returns:
            Dictionary representation of the span
        """
        span_dict = {
            "span_id": format(span.context.span_id, "016x"),
            "trace_id": format(span.context.trace_id, "032x"),
            "name": span.name,
            "kind": str(span.kind) if hasattr(span, "kind") else None,
            "start_time": span.start_time if hasattr(span, "start_time") else None,
            "end_time": span.end_time if hasattr(span, "end_time") else None,
            "attributes": dict(span.attributes) if hasattr(span, "attributes") and span.attributes else {},
            "events": [],
            "status": {
                "status_code": str(span.status.status_code) if hasattr(span, "status") and hasattr(span.status, "status_code") else None,
                "description": span.status.description if hasattr(span, "status") and hasattr(span.status, "description") else None,
            } if hasattr(span, "status") else {},
        }
        
        # Add events if available
        if hasattr(span, "events") and span.events:
            for event in span.events:
                span_dict["events"].append({
                    "name": event.name if hasattr(event, "name") else None,
                    "timestamp": event.timestamp if hasattr(event, "timestamp") else None,
                    "attributes": dict(event.attributes) if hasattr(event, "attributes") and event.attributes else {},
                })
        
        return span_dict
    
    def shutdown(self):
        """Shutdown the exporter and close the file."""
        try:
            with self.file_lock:
                if self._file and not self._file.closed:
                    self._file.close()
        except Exception as e:
            logger.error(f"Error closing file exporter: {e}", exc_info=True)
    
    def force_flush(self, timeout_millis: Optional[int] = None) -> bool:
        """Force flush any buffered spans.
        
        Args:
            timeout_millis: Timeout in milliseconds (not used for file exporter)
            
        Returns:
            True if flush was successful
        """
        try:
            with self.file_lock:
                if self._file and not self._file.closed:
                    self._file.flush()
            return True
        except Exception as e:
            logger.error(f"Error flushing file exporter: {e}", exc_info=True)
            return False
