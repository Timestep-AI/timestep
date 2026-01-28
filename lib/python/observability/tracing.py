"""OpenTelemetry tracing helpers for observability and evaluations.

This module provides utilities for:
- OpenTelemetry tracing setup and configuration
- Creating OTel Test and GenAI semantic convention spans and events
- Baggage propagation
- Metrics instrumentation
- Workflow/task/agent decorators
"""

import logging
import os
from typing import Optional, Dict, Any
from functools import wraps
from opentelemetry import trace, baggage, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry components
try:
    from opentelemetry import trace, baggage, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.debug("OpenTelemetry not available, tracing will be disabled")
    # Create dummy types for type checking
    trace = None
    baggage = None
    metrics = None
    TracerProvider = None
    BatchSpanProcessor = None
    Resource = None
    SERVICE_NAME = None
    MeterProvider = None
    PeriodicExportingMetricReader = None

# Get tracer for observability module (only if available)
if OPENTELEMETRY_AVAILABLE:
    tracer = trace.get_tracer(__name__)
else:
    tracer = None

# Configure logger to suppress noisy OTLP export errors
# This prevents "Transient error StatusCode.UNAVAILABLE" messages from cluttering logs
if OPENTELEMETRY_AVAILABLE:
    export_logger = logging.getLogger("opentelemetry.sdk.trace.export")
    export_logger.setLevel(logging.WARNING)  # Suppress INFO/DEBUG export errors


# High-level tracing API
def enable_tracing(
    app: Optional[Any] = None,
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Enable OpenTelemetry tracing for the application.
    
    This function combines setup and instrumentation into a single call.
    It handles all error cases gracefully.
    
    Args:
        app: FastAPI application instance (optional, can be None if only setting up)
        service_name: Service name for traces (defaults to OTEL_SERVICE_NAME or "timestep")
        otlp_endpoint: OTLP endpoint URL (defaults to OTEL_EXPORTER_OTLP_ENDPOINT or "http://localhost:4317")
        enabled: Whether to enable tracing (defaults to OTEL_ENABLED or True)
    
    Returns:
        True if tracing was successfully enabled, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.debug("OpenTelemetry not available, skipping tracing setup")
        return False
    
    # Setup tracing (internal function)
    if not _setup_tracing(service_name, otlp_endpoint, enabled):
        return False
    
    # Instrument FastAPI app if provided
    if app is not None:
        _instrument_fastapi_app(app)
    
    return True


def disable_tracing() -> None:
    """Disable OpenTelemetry tracing.
    
    This resets the tracer provider to a no-op implementation.
    """
    if not OPENTELEMETRY_AVAILABLE:
        return
    
    try:
        from opentelemetry.trace import NoOpTracerProvider
        trace.set_tracer_provider(NoOpTracerProvider())
        logger.debug("Tracing disabled")
    except Exception as e:
        logger.debug(f"Error disabling tracing: {e}")


# Internal tracing setup and configuration
def _setup_tracing(
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
        
        # Log initialization (warn if OTLP endpoint might be unavailable)
        logger.info(
            f"OpenTelemetry tracing initialized: service={service_name}, "
            f"otlp_endpoint={otlp_endpoint}"
        )
        
        # Note: Export errors are suppressed via logger configuration above
        # If OTLP is unavailable, spans will be dropped silently after initial warning
        
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
    
    Note: FastAPI instrumentation is done separately via _instrument_fastapi_app()
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


def _instrument_fastapi_app(app):
    """Instrument a FastAPI application for tracing (internal helper).
    
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


# Baggage propagation utilities
def set_baggage(key: str, value: str) -> None:
    """Set a baggage value in the current context.
    
    Args:
        key: Baggage key (e.g., "test.suite.name")
        value: Baggage value
    """
    if not OPENTELEMETRY_AVAILABLE:
        return
    
    try:
        baggage.set_baggage(key, value)
    except Exception as e:
        logger.debug(f"Error setting baggage {key}={value}: {e}")


def get_baggage(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a baggage value from the current context.
    
    Args:
        key: Baggage key (e.g., "test.suite.name")
        default: Default value if key not found
    
    Returns:
        Baggage value or default
    """
    if not OPENTELEMETRY_AVAILABLE:
        return default
    
    try:
        return baggage.get_baggage(key) or default
    except Exception:
        return default


def set_test_context(suite_name: Optional[str] = None, case_name: Optional[str] = None) -> None:
    """Set test context in baggage for eval runs/cases.
    
    Args:
        suite_name: Test suite name (eval spec name/ID)
        case_name: Test case name (dataset/item identifier)
    """
    if suite_name:
        set_baggage("test.suite.name", suite_name)
    if case_name:
        set_baggage("test.case.name", case_name)


def get_test_context() -> Dict[str, Optional[str]]:
    """Get test context from baggage.
    
    Returns:
        Dict with "suite_name" and "case_name" keys
    """
    return {
        "suite_name": get_baggage("test.suite.name"),
        "case_name": get_baggage("test.case.name"),
    }


# Metrics instrumentation helpers
def setup_metrics(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
) -> bool:
    """Initialize OpenTelemetry metrics with OTLP exporter.
    
    Args:
        service_name: Service name for metrics (defaults to OTEL_SERVICE_NAME)
        otlp_endpoint: OTLP endpoint URL (defaults to OTEL_EXPORTER_OTLP_ENDPOINT)
    
    Returns:
        True if metrics were successfully initialized, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        return False
    
    try:
        # Check if already initialized
        if isinstance(metrics.get_meter_provider(), MeterProvider):
            logger.debug("Metrics already initialized, skipping")
            return True
        
        if service_name is None:
            service_name = os.getenv("OTEL_SERVICE_NAME", "timestep")
        
        if otlp_endpoint is None:
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        
        # Create resource
        resource = Resource.create({SERVICE_NAME: service_name})
        
        # Create OTLP metric exporter
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
        except ImportError:
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
            except ImportError:
                logger.warning("OTLP metric exporter not available")
                return False
        
        # Create metric reader
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
        
        # Create meter provider
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        
        logger.info(f"OpenTelemetry metrics initialized: service={service_name}, otlp_endpoint={otlp_endpoint}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry metrics: {e}", exc_info=True)
        return False


def get_meter(name: str = __name__):
    """Get a meter for recording metrics.
    
    Args:
        name: Meter name (defaults to module name)
    
    Returns:
        Meter instance or None if metrics not available
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None
    
    try:
        return metrics.get_meter(name)
    except Exception:
        return None


# Workflow/Task/Agent decorators (pure OTel, FOSS)
def workflow(name: str):
    """Decorator to mark a function as a workflow (pure OTel).
    
    Creates a span with gen_ai.operation.name = "workflow".
    
    Args:
        name: Workflow name
    
    Example:
        @workflow(name="joke_creation")
        def create_joke():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"workflow {name}",
                kind=trace.SpanKind.INTERNAL,
                attributes={
                    "gen_ai.operation.name": "workflow",
                    "gen_ai.workflow.name": name,  # Custom attribute (not in OTel spec yet)
                }
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def task(name: str):
    """Decorator to mark a function as a task (pure OTel).
    
    Creates a span with gen_ai.operation.name = "task".
    
    Args:
        name: Task name
    
    Example:
        @task(name="generate_response")
        def generate_response():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"task {name}",
                kind=trace.SpanKind.INTERNAL,
                attributes={
                    "gen_ai.operation.name": "task",  # Custom, or use existing OTel op names
                    "gen_ai.task.name": name,  # Custom attribute
                }
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def agent_span(name: str, agent_id: Optional[str] = None):
    """Decorator to mark a function as an agent invocation (pure OTel GenAI).
    
    Creates a span with gen_ai.operation.name = "invoke_agent".
    
    Args:
        name: Agent name
        agent_id: Optional agent ID
    
    Example:
        @agent_span(name="personal_assistant")
        def handle_request():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = trace.get_tracer(__name__)
            attrs = {
                "gen_ai.operation.name": "invoke_agent",  # OTel GenAI standard
                "gen_ai.agent.name": name,  # OTel GenAI standard
            }
            if agent_id:
                attrs["gen_ai.agent.id"] = agent_id  # OTel GenAI standard
            
            with tracer.start_as_current_span(
                f"invoke_agent {name}",
                kind=trace.SpanKind.INTERNAL,
                attributes=attrs
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Eval-specific span creation and helpers
def create_test_suite_span(
    suite_name: str,
    run_status: Optional[str] = None,
) -> trace.Span:
    """Create a test.suite span for an eval run.
    
    Args:
        suite_name: Test suite name (eval spec name or ID)
        run_status: Overall run status ("pass", "fail", "skip", "error")
    
    Returns:
        OTel span with test.suite.* attributes
    """
    span = tracer.start_span(
        "test.suite",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "test.suite.name": suite_name,
        }
    )
    
    if run_status:
        span.set_attribute("test.suite.run.status", run_status)
    
    return span


def create_test_case_span(
    case_name: str,
    result_status: Optional[str] = None,
    parent: Optional[trace.Span] = None,
) -> trace.Span:
    """Create a test.case span for a single eval case.
    
    Args:
        case_name: Test case name (dataset/item identifier)
        result_status: Case result status ("pass", "fail", "skip")
        parent: Parent span (test.suite span)
    
    Returns:
        OTel span with test.case.* attributes
    """
    span = tracer.start_span(
        "test.case",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "test.case.name": case_name,
        },
        parent=parent,
    )
    
    if result_status:
        span.set_attribute("test.case.result.status", result_status)
    
    return span


def create_invoke_agent_span(
    agent_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    parent: Optional[trace.Span] = None,
):
    """Create an invoke_agent span context manager for an agent step.
    
    Args:
        agent_name: Agent name (optional)
        agent_id: Agent ID (optional)
        parent: Parent span (test.case span)
    
    Returns:
        Context manager for OTel span with gen_ai.agent.* attributes
    """
    span_name = f"invoke_agent {agent_name}" if agent_name else "invoke_agent"
    
    attrs = {
        "gen_ai.operation.name": "invoke_agent",
    }
    
    if agent_id:
        attrs["gen_ai.agent.id"] = agent_id
    if agent_name:
        attrs["gen_ai.agent.name"] = agent_name
    
    # Use start_as_current_span to get a context manager
    return tracer.start_as_current_span(
        span_name,
        kind=trace.SpanKind.INTERNAL,
        attributes=attrs,
    )


def emit_evaluation_result_event(
    span: trace.Span,
    evaluation_name: str,
    score_value: Optional[float] = None,
    score_label: Optional[str] = None,
    explanation: Optional[str] = None,
    response_id: Optional[str] = None,
    error_type: Optional[str] = None,
) -> None:
    """Emit a gen_ai.evaluation.result event on a span.
    
    Args:
        span: The span to attach the event to (typically a GenAI operation span)
        evaluation_name: Name of the evaluation metric (e.g., "accuracy", "relevance")
        score_value: Numeric score (0.0-1.0 or other scale)
        score_label: Human-readable label ("pass", "fail", "partial", etc.)
        explanation: Optional explanation for the score
        response_id: Optional response ID to correlate with operation span
        error_type: Optional error type if evaluation failed
    """
    attrs: Dict[str, Any] = {
        "gen_ai.evaluation.name": evaluation_name,
    }
    
    if score_value is not None:
        attrs["gen_ai.evaluation.score.value"] = float(score_value)
    
    if score_label:
        attrs["gen_ai.evaluation.score.label"] = score_label
    
    if explanation:
        attrs["gen_ai.evaluation.explanation"] = explanation
    
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    
    if error_type:
        attrs["error.type"] = error_type
    
    span.add_event("gen_ai.evaluation.result", attributes=attrs)


def set_test_case_status(
    span: trace.Span,
    status: str,
) -> None:
    """Set test.case.result.status attribute on a test.case span.
    
    Args:
        span: The test.case span
        status: Status value ("pass", "fail", "skip")
    """
    span.set_attribute("test.case.result.status", status)
    
    # Also set span status based on result
    if status == "pass":
        span.set_status(Status(StatusCode.OK))
    elif status == "fail":
        span.set_status(Status(StatusCode.ERROR))
    elif status == "skip":
        span.set_status(Status(StatusCode.OK))  # Skip is OK, just not executed


def set_test_suite_status(
    span: trace.Span,
    status: str,
) -> None:
    """Set test.suite.run.status attribute on a test.suite span.
    
    Args:
        span: The test.suite span
        status: Status value ("pass", "fail", "skip", "error")
    """
    span.set_attribute("test.suite.run.status", status)
    
    # Also set span status based on result
    if status == "pass":
        span.set_status(Status(StatusCode.OK))
    elif status in ("fail", "error"):
        span.set_status(Status(StatusCode.ERROR))
    elif status == "skip":
        span.set_status(Status(StatusCode.OK))  # Skip is OK, just not executed
