"""Observability module - OpenTelemetry tracing integration."""

from timestep.observability.tracing import setup_tracing, get_tracer

__all__ = ["setup_tracing", "get_tracer"]
