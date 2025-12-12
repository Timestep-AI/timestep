"""Tests for basic agent functionality."""

from unittest.mock import AsyncMock, patch

import pytest
from timestep import run_agent, GetWeather


@pytest.mark.asyncio
async def test_basic_agent_response():
    """Test basic agent response without tools."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's 2+2?"},
    ]
    
    response = await run_agent(messages)
    
    assert response is not None
    assert len(response) > 0
    # Should contain "4" in the response
    assert "4" in response


@pytest.mark.asyncio
async def test_agent_with_tool():
    """Test agent using a tool."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that can answer questions about weather. When asked about weather, you MUST use the getWeather tool.",
        },
        {"role": "user", "content": "What's the weather in Oakland?"},
    ]
    
    tools = [GetWeather]
    
    # Mock the MCP tool call
    with patch("timestep.core.tools.call_mcp_tool", new_callable=AsyncMock) as mock_mcp_tool:
        mock_mcp_tool.return_value = "The weather in Oakland is sunny"
        
        response = await run_agent(messages, tools=tools)
        
        assert response is not None
        assert len(response) > 0
        # Should mention Oakland or weather
        assert "Oakland" in response or "weather" in response.lower()
        
        # Verify MCP tool was called
        mock_mcp_tool.assert_called_once_with("get_weather", {"city": "Oakland"})


@pytest.mark.asyncio
async def test_agent_conversation():
    """Test agent maintaining conversation context."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's 2+2?"},
    ]
    
    # First message (run_agent will append the assistant response to messages)
    response1 = await run_agent(messages)
    assert "4" in response1
    
    # Follow-up message with history (should remember)
    # Note: run_agent already appended the assistant message, so we just add the user message
    messages.append({"role": "user", "content": "What's three times that number?"})
    response2 = await run_agent(messages)
    
    assert response2 is not None
    # Should mention 12 or the calculation
    assert "12" in response2 or "three" in response2.lower()
