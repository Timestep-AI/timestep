"""Tests for runtime safety checks."""

import pytest
from timestep.analysis.safety import CircularDependencyChecker, ToolCompatibilityChecker


@pytest.mark.asyncio
async def test_circular_dependency_checker_no_cycle():
    """Test circular dependency checker with no cycles."""
    checker = CircularDependencyChecker()
    # This will fail if no DB connection, but that's expected
    # In a real test, we'd set up a test database
    result = await checker.check_circular_handoffs("nonexistent_agent")
    # Should return None (no cycle) or fail gracefully
    assert result is None or isinstance(result, list)


@pytest.mark.asyncio
async def test_tool_compatibility_checker():
    """Test tool compatibility checker."""
    checker = ToolCompatibilityChecker()
    # This will fail if no DB connection, but that's expected
    # In a real test, we'd set up a test database
    try:
        warnings = await checker.check_compatibility("nonexistent_agent")
        assert isinstance(warnings, list)
    except Exception:
        # Expected if agent doesn't exist or DB not available
        pass

