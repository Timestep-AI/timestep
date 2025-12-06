"""Compatibility layer for importing vendored or installed agents package."""

import sys
import types
import importlib

# Try to import from vendored code first (for published packages), 
# then fall back to installed package (for development)
try:
    # Add sys.modules shim so vendored code's absolute imports work
    # This must happen BEFORE importing from _vendored.agents
    # We need to pre-populate sys.modules with the vendored submodules
    # that the vendored code's __init__.py will try to import
    
    # Create the base 'agents' module first
    _agents_proxy = types.ModuleType('agents')
    _agents_proxy.__path__ = []
    sys.modules['agents'] = _agents_proxy
    
    # Pre-populate key submodules that are imported during module initialization
    # Import them from the vendored location and register them in sys.modules
    try:
        sys.modules['agents.model_settings'] = importlib.import_module('timestep._vendored.agents.model_settings')
    except ImportError:
        pass
    
    # Now import the main vendored module - its internal imports will find 'agents' in sys.modules
    import timestep._vendored.agents as _vendored_agents_module
    
    # Replace the proxy with the real module
    sys.modules['agents'] = _vendored_agents_module
    
    from ._vendored.agents import (
        Agent, Runner, RunConfig, RunState, TResponseInputItem,
        Model, ModelProvider, ModelResponse, Usage, ModelSettings, 
        ModelTracing, Handoff, Tool, RawResponsesStreamEvent,
        function_tool, OpenAIProvider,
        input_guardrail, output_guardrail, GuardrailFunctionOutput,
        RunContextWrapper, OpenAIConversationsSession
    )
    from ._vendored.agents.exceptions import (
        AgentsException, MaxTurnsExceeded, ModelBehaviorError, UserError,
        InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
    )
    from ._vendored.agents.memory.session import SessionABC
except ImportError:
    # Fall back to installed package (development mode)
    from agents import (
        Agent, Runner, RunConfig, RunState, TResponseInputItem,
        Model, ModelProvider, ModelResponse, Usage, ModelSettings,
        ModelTracing, Handoff, Tool, RawResponsesStreamEvent,
        function_tool, OpenAIProvider,
        input_guardrail, output_guardrail, GuardrailFunctionOutput,
        RunContextWrapper, OpenAIConversationsSession
    )
    from agents.exceptions import (
        AgentsException, MaxTurnsExceeded, ModelBehaviorError, UserError,
        InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
    )
    from agents.memory.session import SessionABC

__all__ = [
    'Agent', 'Runner', 'RunConfig', 'RunState', 'TResponseInputItem',
    'Model', 'ModelProvider', 'ModelResponse', 'Usage', 'ModelSettings',
    'ModelTracing', 'Handoff', 'Tool', 'RawResponsesStreamEvent',
    'function_tool', 'OpenAIProvider',
    'input_guardrail', 'output_guardrail', 'GuardrailFunctionOutput',
    'RunContextWrapper', 'OpenAIConversationsSession',
    'AgentsException', 'MaxTurnsExceeded', 'ModelBehaviorError', 'UserError',
    'InputGuardrailTripwireTriggered', 'OutputGuardrailTripwireTriggered',
    'SessionABC',
]

