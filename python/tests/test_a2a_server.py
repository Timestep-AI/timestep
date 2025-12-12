"""Tests for A2A server setup."""

import pytest
from timestep.a2a.server import create_agent_card, create_server


def test_create_skills_from_tools():
    """Test creating AgentSkill objects from tools."""
    from timestep.a2a.postgres_agent_store import Agent
    
    # Create a test agent with both tools
    test_agent = Agent(
        id="test-agent",
        name="Test Agent",
        description="Test agent",
        tools=["get_weather", "web_search"],
        model="gpt-4.1",
    )
    
    # Test by checking the agent card skills
    card = create_agent_card(test_agent)
    skills = card.skills
    
    assert len(skills) == 2
    
    # Find skills by ID (order may vary)
    weather_skill = next((s for s in skills if s.id == "get_weather"), None)
    search_skill = next((s for s in skills if s.id == "web_search"), None)
    
    assert weather_skill is not None
    assert weather_skill.name == "Get Weather"
    assert len(weather_skill.examples) > 0
    
    assert search_skill is not None
    assert search_skill.name == "Web Search"
    assert len(search_skill.examples) > 0


def test_create_agent_card():
    """Test creating AgentCard."""
    from timestep.a2a.postgres_agent_store import Agent
    
    url = "http://localhost:8080/"
    test_agent = Agent(
        id="test-agent",
        name="Test Agent",
        description="Test agent",
        tools=["get_weather", "web_search"],
        model="gpt-4.1",
    )
    
    card = create_agent_card(test_agent, url)
    
    assert card.name == "Test Agent"
    assert card.url == url
    assert card.version == "2026.0.5"
    assert card.default_input_modes == ["text"]
    assert "text" in card.default_output_modes
    assert "task-status" in card.default_output_modes
    assert card.capabilities.streaming is True
    assert len(card.skills) == 2
    # Examples may not be a direct attribute, check if it exists
    if hasattr(card, "examples"):
        assert len(card.examples) > 0


def test_create_server():
    """Test creating A2A server."""
    app = create_server(host="127.0.0.1", port=8080)
    
    # Verify server was created (should be a Starlette app)
    assert app is not None
    # Check that it has routes (Starlette apps have routes)
    assert hasattr(app, "routes") or hasattr(app, "router")


def test_create_server_with_custom_tools():
    """Test creating server with custom tools."""
    from timestep.core import GetWeather
    
    app = create_server(host="127.0.0.1", port=8080, tools=[GetWeather])
    assert app is not None


def test_create_server_with_custom_model():
    """Test creating server with custom model."""
    app = create_server(host="127.0.0.1", port=8080, model="gpt-4")
    assert app is not None


def test_create_agent_card_with_default_url():
    """Test creating agent card with default URL."""
    from timestep.a2a.postgres_agent_store import Agent
    
    test_agent = Agent(
        id="test-agent",
        name="Test Agent",
        description="Test agent",
        tools=["get_weather"],
        model="gpt-4.1",
    )
    
    card = create_agent_card(test_agent)
    assert card.url == "http://localhost:8080/"

