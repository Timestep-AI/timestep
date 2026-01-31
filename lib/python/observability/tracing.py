"""OpenTelemetry tracing helpers for observability."""

import logging
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

# Optional: LoggerProvider for OTel logs (used by OpenAI instrumentation for message content)
# SDK log components (LoggerProvider, ConsoleLogExporter) are in opentelemetry-sdk
try:
    from opentelemetry import _logs
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        ConsoleLogExporter,
    )
    _SDK_LOGS_AVAILABLE = True
except ImportError:
    _logs = None
    LoggerProvider = None
    BatchLogRecordProcessor = None
    ConsoleLogExporter = None
    _SDK_LOGS_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    _OTLP_LOGS_AVAILABLE = True
except ImportError:
    OTLPLogExporter = None
    _OTLP_LOGS_AVAILABLE = False

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

try:
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
except ImportError:
    RequestsInstrumentor = None

try:
    from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
except ImportError:
    URLLib3Instrumentor = None

try:
    from opentelemetry.instrumentation.urllib import URLLibInstrumentor
except ImportError:
    URLLibInstrumentor = None

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
    - OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: Set to 'true' to capture prompts/completions as OTel log events (requires LoggerProvider)
    - OTEL_LOGS_EXPORTER: Log export target(s), comma-separated: 'console' (stdout), 'otlp', 'file'. Default: 'otlp' if available, else 'console'
    - OTEL_LOGS_FILE_PATH: Path for file exporter (default: data/otel_logs.jsonl). Used when OTEL_LOGS_EXPORTER contains 'file'.

    Args:
        app: Optional FastAPI app instance to instrument. If provided, instruments this specific app
             (excluding only /.well-known). If None, does not instrument FastAPI.
    """
    # Check if already initialized
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        logger.debug("Tracing already initialized")
        
        # Check message content capture status even if tracing is already initialized
        capture_message_content = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false").lower() == "true"
        if capture_message_content:
            logger.info("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true - message content will be captured as OTel log events (if LoggerProvider is configured)")
        else:
            logger.info("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT not set or false - message content will not be captured")

        # If app is provided and FastAPI instrumentation is available, instrument it
        if app is not None and FastAPIInstrumentor is not None:
            try:
                excluded_urls = "/.well-known"
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

    # Configure LoggerProvider so OpenAI instrumentation can emit message content as OTel logs.
    # The instrumentation uses logger.emit(LogRecord(...)) for prompts/completions when
    # OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true. Without a LoggerProvider,
    # those log records are never exported (Jaeger "Events" tab shows span events, not logs).
    # OTEL_LOGS_EXPORTER: 'console' (stdout), 'otlp', 'file', or comma-separated (e.g. 'console,file').
    if _SDK_LOGS_AVAILABLE:
        try:
            exporters_env = os.getenv("OTEL_LOGS_EXPORTER", "otlp" if _OTLP_LOGS_AVAILABLE else "console").strip().lower()
            exporters_list = [x.strip() for x in exporters_env.split(",") if x.strip()]
            if not exporters_list:
                exporters_list = ["console"]
            logger_provider = LoggerProvider(resource=resource)
            added = []
            if "console" in exporters_list:
                logger_provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogExporter()))
                added.append("console")
            if "file" in exporters_list:
                try:
                    from timestep.observability.file_log_exporter import FileLogExporter
                    file_path = os.getenv("OTEL_LOGS_FILE_PATH", "data/otel_logs.jsonl").strip()
                    if file_path:
                        logger_provider.add_log_record_processor(BatchLogRecordProcessor(FileLogExporter(file_path)))
                        added.append("file")
                    else:
                        logger.warning("OTEL_LOGS_EXPORTER=file requested but OTEL_LOGS_FILE_PATH is empty")
                except Exception as file_exc:
                    logger.warning("Could not add file log exporter: %s", file_exc)
            if "otlp" in exporters_list and _OTLP_LOGS_AVAILABLE:
                logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
                added.append("otlp")
            elif "otlp" in exporters_list and not _OTLP_LOGS_AVAILABLE:
                logger.warning("OTEL_LOGS_EXPORTER=otlp requested but opentelemetry-exporter-otlp-proto-http not available; use OTEL_LOGS_EXPORTER=console for stdout")
            if added:
                _logs.set_logger_provider(logger_provider)
                logger.info("LoggerProvider configured (exporters: %s) - message content as OTel logs when OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true", ",".join(added))
        except Exception as e:
            logger.warning("Could not configure LoggerProvider for message content capture: %s", e)
    else:
        logger.debug("LoggerProvider not configured - opentelemetry-sdk required for log export")

    # Instrument OpenAI
    # The instrumentation automatically reads OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT from environment
    # IMPORTANT: The environment variable must be set BEFORE importing the instrumentation
    # Check if message content capture is enabled
    capture_message_content = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false").lower() == "true"
    if capture_message_content:
        logger.info("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true - message content will be captured as OTel log events (if LoggerProvider is configured)")
    else:
        logger.info("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT not set or false - message content will not be captured")
        logger.info("Set OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true to capture prompts and completions as log events")
    
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

    # Instrument requests so outbound HTTP from tools (e.g. web_search/DDGS) are child spans
    if RequestsInstrumentor is not None:
        try:
            RequestsInstrumentor().instrument()
            logger.info("requests instrumentation enabled - tool HTTP calls (e.g. DuckDuckGo) will be child spans")
        except Exception as e:
            logger.warning("Failed to instrument requests: %s", e)

    # Instrument urllib3/urllib so any tool using them (e.g. ddgs/primp) gets context propagation.
    # Optional: if instrumentors not installed, those HTTP calls may appear as root traces.
    if URLLib3Instrumentor is not None:
        try:
            URLLib3Instrumentor().instrument()
            logger.info("urllib3 instrumentation enabled - tool HTTP calls will be child spans")
        except Exception as e:
            logger.warning("Failed to instrument urllib3: %s", e)
    if URLLibInstrumentor is not None:
        try:
            URLLibInstrumentor().instrument()
            logger.info("urllib instrumentation enabled - tool HTTP calls will be child spans")
        except Exception as e:
            logger.warning("Failed to instrument urllib: %s", e)

    # Instrument FastAPI app if provided, excluding only non-semantic routes
    if app is not None and FastAPIInstrumentor is not None:
        try:
            # Exclude only /.well-known (agent card). Do NOT exclude /mcp: tool execution
            # (e.g. web_search -> DDGS) runs in the /mcp request handler; if /mcp is excluded,
            # that handler has no trace context, so outbound calls (e.g. to search backends)
            # create new root traces instead of being children of the same request trace.
            excluded_urls = "/.well-known"
            FastAPIInstrumentor().instrument_app(app, excluded_urls=excluded_urls)
            logger.info(f"FastAPI app instrumented (excluding: {excluded_urls})")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI app: {e}")
    
    logger.info("OpenTelemetry tracing configured")
