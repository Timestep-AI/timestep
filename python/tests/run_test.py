#!/usr/bin/env python3
"""
Test runner for Python implementation.
Accepts a test case JSON file and executes it.
"""

import json
import sys
import os
import re
import asyncio
from pathlib import Path
from typing import Dict, Any
import time

# Add parent directory to path to import timestep
sys.path.insert(0, str(Path(__file__).parent.parent))

from timestep import MultiModelProvider, MultiModelProviderMap, OllamaModelProvider

try:
    from agents import Agent, Runner, RunConfig, ModelSettings
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


async def run_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a test case and return the result."""
    start_time = time.time()
    
    try:
        # Setup provider based on test case
        setup = test_case.get("setup", {})
        provider_type = setup.get("provider_type", "multi")
        provider_config = setup.get("provider_config", {})
        
        if provider_type == "ollama":
            provider = OllamaModelProvider(
                api_key=provider_config.get("api_key"),
                base_url=provider_config.get("base_url"),
            )
        elif provider_type == "multi":
            provider_map = MultiModelProviderMap()
            if provider_config.get("ollama_api_key"):
                provider_map.add_provider(
                    "ollama",
                    OllamaModelProvider(api_key=provider_config.get("ollama_api_key"))
                )
            # Use environment variable if test config has placeholder, otherwise use config value
            openai_api_key = provider_config.get("openai_api_key")
            if openai_api_key == "test-key" or not openai_api_key:
                openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            
            # Build provider kwargs with optional parameters
            provider_kwargs = {
                "provider_map": provider_map,
                "openai_api_key": openai_api_key,
            }
            
            # Add optional OpenAI provider options
            if provider_config.get("openai_base_url") is not None:
                provider_kwargs["openai_base_url"] = provider_config.get("openai_base_url")
            if provider_config.get("openai_organization") is not None:
                provider_kwargs["openai_organization"] = provider_config.get("openai_organization")
            if provider_config.get("openai_project") is not None:
                provider_kwargs["openai_project"] = provider_config.get("openai_project")
            if provider_config.get("openai_use_responses") is not None:
                provider_kwargs["openai_use_responses"] = provider_config.get("openai_use_responses")
            
            provider = MultiModelProvider(**provider_kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        # Get input
        test_input = test_case.get("input", {})
        model_name = test_input.get("model_name")
        run_agent = test_input.get("run_agent", False)
        stream = test_input.get("stream", False)
        
        actual_result = {
            "model_name": model_name,
            "provider_type": provider_type,
        }
        assertion_failures = []
        agent_output = None
        
        if run_agent:
            if not AGENTS_AVAILABLE:
                raise ImportError("agents package is required for agent tests. Install with: pip install openai-agents")
            
            # Create agent configuration
            agent_config = test_input.get("agent_config") or {}
            # Agent constructor requires 'name' parameter
            agent_kwargs = {
                "name": test_case.get("name", "test_agent"),
                "model": model_name
            }
            
            if agent_config and agent_config.get("system_prompt"):
                agent_kwargs["system"] = agent_config["system_prompt"]
            
            # Set temperature to 0 via ModelSettings
            temperature = agent_config.get("temperature", 0) if agent_config and agent_config.get("temperature") is not None else 0
            model_settings = ModelSettings(temperature=temperature)
            agent_kwargs["model_settings"] = model_settings
            
            # Create agent
            agent = Agent(**agent_kwargs)
            
            # Get user input
            user_input = test_input.get("user_input")
            if user_input is None:
                # Fallback to messages if user_input not provided
                messages = test_input.get("messages", [])
                if messages and len(messages) > 0:
                    first_msg = messages[0]
                    if isinstance(first_msg, dict):
                        user_input = first_msg.get("content", "Hello")
                    else:
                        user_input = str(first_msg) if first_msg else "Hello"
                else:
                    user_input = "Hello"
            
            # Convert user_input to string if it's a dict
            if isinstance(user_input, dict):
                user_input = user_input.get("content", str(user_input))
            elif not isinstance(user_input, str):
                user_input = str(user_input)
            
            # Create run config
            run_config = RunConfig(model_provider=provider)
            
            # Run agent (streaming or non-streaming)
            try:
                if stream:
                    # Use streaming mode - run_streamed returns a RunResultStreaming object
                    agent_output = ""
                    try:
                        stream_result = Runner.run_streamed(agent, user_input, run_config=run_config)
                        
                        # Try to get the stream from the result
                        # Based on error message, RunResultStreaming has 'stream_events' method
                        stream_iter = None
                        if hasattr(stream_result, 'stream_events'):
                            # stream_events is a method, so call it
                            stream_iter = stream_result.stream_events()
                        elif hasattr(stream_result, '__aiter__'):
                            stream_iter = stream_result
                        elif hasattr(stream_result, 'stream'):
                            stream_prop = stream_result.stream
                            # Check if it's a method or property
                            if callable(stream_prop):
                                stream_iter = stream_prop()
                            else:
                                stream_iter = stream_prop
                        elif hasattr(stream_result, 'events'):
                            events_prop = stream_result.events
                            if callable(events_prop):
                                stream_iter = events_prop()
                            else:
                                stream_iter = events_prop
                        elif callable(stream_result):
                            stream_iter = stream_result()
                        else:
                            # Try to get final_output as fallback
                            if hasattr(stream_result, 'final_output') and stream_result.final_output:
                                agent_output = str(stream_result.final_output)
                            else:
                                # Debug: print available attributes
                                attrs = [a for a in dir(stream_result) if not a.startswith('_')]
                                raise RuntimeError(f"Unable to iterate over RunResultStreaming: {type(stream_result)}, available attributes: {attrs}")
                        
                        # Iterate over the stream and collect text deltas
                        if stream_iter:
                            async for chunk in stream_iter:
                                # Extract delta/text from various possible structures
                                delta = None
                                
                                # Check for delta attribute
                                if hasattr(chunk, 'delta') and chunk.delta:
                                    delta = str(chunk.delta)
                                # Check for delta in dict
                                elif isinstance(chunk, dict):
                                    if 'delta' in chunk:
                                        delta = str(chunk['delta'])
                                    elif 'text' in chunk:
                                        delta = str(chunk['text'])
                                    elif 'content' in chunk:
                                        content = chunk['content']
                                        if isinstance(content, list):
                                            delta = ' '.join(str(c.get('text', c) if isinstance(c, dict) else c) for c in content)
                                        else:
                                            delta = str(content)
                                # Check for text attribute
                                elif hasattr(chunk, 'text') and chunk.text:
                                    delta = str(chunk.text)
                                # Check for content attribute
                                elif hasattr(chunk, 'content'):
                                    content = chunk.content
                                    if isinstance(content, list):
                                        delta = ' '.join(str(c.get('text', c) if isinstance(c, dict) else c) for c in content)
                                    else:
                                        delta = str(content)
                                
                                if delta:
                                    agent_output += delta
                            
                            # If we got nothing from deltas, try final_output
                            if not agent_output and hasattr(stream_result, 'final_output'):
                                final_output = stream_result.final_output
                                if final_output is not None:
                                    agent_output = str(final_output)
                                
                    except Exception as stream_error:
                        raise RuntimeError(f"Streaming failed: {str(stream_error)}") from stream_error
                    
                    # Ensure agent_output is not None or empty string "None"
                    if agent_output is None or agent_output == "None":
                        agent_output = ""
                else:
                    # Non-streaming mode
                    result = await Runner.run(agent, user_input, run_config=run_config)
                    
                    # Extract output from result (only for non-streaming)
                    # The result structure depends on the agents library version
                    if hasattr(result, 'final_output') and result.final_output:
                        # Try final_output first (most direct)
                        agent_output = str(result.final_output)
                    elif hasattr(result, 'messages') and result.messages:
                        # Get the last assistant message
                        last_message = result.messages[-1]
                        if hasattr(last_message, 'content'):
                            content = last_message.content
                            # content might be a list or string
                            if isinstance(content, list):
                                agent_output = ' '.join(str(c.get('text', c) if isinstance(c, dict) else c) for c in content)
                            else:
                                agent_output = str(content)
                        elif isinstance(last_message, dict):
                            agent_output = last_message.get("content", str(last_message))
                        else:
                            agent_output = str(last_message)
                    elif hasattr(result, 'content'):
                        agent_output = result.content
                    elif isinstance(result, str):
                        agent_output = result
                    else:
                        # Try to extract from string representation
                        result_str = str(result)
                        # Look for "Final output" pattern in Python RunResult
                        if "Final output" in result_str:
                            # Extract text between "Final output" and next section
                            match = re.search(r'Final output.*?:\s*(.+?)(?:\n-|\Z)', result_str, re.DOTALL)
                            if match:
                                agent_output = match.group(1).strip()
                            else:
                                agent_output = result_str
                        else:
                            agent_output = result_str
                
                if agent_output is not None and agent_output != "None":
                    actual_result["agent_output"] = agent_output
                
            except Exception as e:
                raise RuntimeError(f"Agent execution failed: {str(e)}") from e
        else:
            # Just verify model creation (original behavior)
            model = provider.get_model(model_name)
            # Model created successfully, no further action needed
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Check expected results
        expected = test_case.get("expected") or {}
        
        # Validate model name
        if expected and expected.get("model_name") and model_name != expected.get("model_name"):
            assertion_failures.append({
                "assertion": {"field": "model_name", "operator": "equals", "value": expected.get("model_name")},
                "actual_value": model_name,
                "reason": f"Expected model_name {expected.get('model_name')}, got {model_name}"
            })
        
        # Validate agent output if provided
        if run_agent and agent_output is not None:
            agent_output_expectation = (expected.get("agent_output") or {}) if expected else {}
            
            # Check contains_text (case-insensitive)
            if agent_output_expectation.get("contains_text"):
                for text in agent_output_expectation["contains_text"]:
                    if text.lower() not in agent_output.lower():
                        assertion_failures.append({
                            "assertion": {"field": "agent_output", "operator": "contains", "value": text},
                            "actual_value": agent_output,
                            "reason": f"Expected agent output to contain '{text}', but it didn't"
                        })
            
            # Check excludes_text
            if agent_output_expectation.get("excludes_text"):
                for text in agent_output_expectation["excludes_text"]:
                    if text in agent_output:
                        assertion_failures.append({
                            "assertion": {"field": "agent_output", "operator": "excludes", "value": text},
                            "actual_value": agent_output,
                            "reason": f"Expected agent output to not contain '{text}', but it did"
                        })
            
            # Check min_length
            if agent_output_expectation.get("min_length") is not None:
                min_len = agent_output_expectation["min_length"]
                if len(agent_output) < min_len:
                    assertion_failures.append({
                        "assertion": {"field": "agent_output", "operator": "min_length", "value": min_len},
                        "actual_value": len(agent_output),
                        "reason": f"Expected agent output length >= {min_len}, got {len(agent_output)}"
                    })
            
            # Check max_length
            max_len = agent_output_expectation.get("max_length")
            if max_len is not None and max_len is not False:  # Handle None and False
                if len(agent_output) > max_len:
                    assertion_failures.append({
                        "assertion": {"field": "agent_output", "operator": "max_length", "value": max_len},
                        "actual_value": len(agent_output),
                        "reason": f"Expected agent output length <= {max_len}, got {len(agent_output)}"
                    })
            
            # Check exact_match (rarely used)
            if agent_output_expectation.get("exact_match"):
                expected_text = agent_output_expectation["exact_match"]
                if agent_output != expected_text:
                    assertion_failures.append({
                        "assertion": {"field": "agent_output", "operator": "equals", "value": expected_text},
                        "actual_value": agent_output,
                        "reason": f"Expected exact match, but got different output"
                    })
        
        status = "Passed" if not assertion_failures else "Failed"
        
        return {
            "test_name": test_case.get("name", "unknown"),
            "implementation": "python",
            "status": status,
            "duration_ms": duration_ms,
            "error": None,
            "actual_result": actual_result,
            "assertion_failures": assertion_failures,
        }
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "test_name": test_case.get("name", "unknown"),
            "implementation": "python",
            "status": "Error",
            "duration_ms": duration_ms,
            "error": str(e),
            "actual_result": None,
            "assertion_failures": [],
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Read from stdin
        test_case_json = sys.stdin.read()
    else:
        # Read from file
        with open(sys.argv[1], "r") as f:
            test_case_json = f.read()
    
    test_case = json.loads(test_case_json)
    result = asyncio.run(run_test(test_case))
    print(json.dumps(result))

