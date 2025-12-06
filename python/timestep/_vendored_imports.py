"""Compatibility layer for importing vendored or installed agents package."""

import sys
import types
import importlib
import importlib.util
from pathlib import Path

# Try to import from vendored code first (for published packages), 
# then fall back to installed package (for development)
try:
    # Add sys.modules shim so vendored code's absolute imports work
    # This must happen BEFORE importing from _vendored.agents
    # We need to pre-populate sys.modules with the vendored submodules
    # that the vendored code's __init__.py will try to import
    
    # Create the base 'agents' module first as a package
    _agents_proxy = types.ModuleType('agents')
    _agents_proxy.__path__ = []
    _agents_proxy.__package__ = 'agents'
    sys.modules['agents'] = _agents_proxy
    
    # Pre-populate key submodules that are imported during module initialization
    # Load them directly from file system to avoid import resolution issues
    try:
        # Get the path to the model_settings module
        _vendored_path = Path(__file__).parent / '_vendored' / 'agents'
        _model_settings_file = _vendored_path / 'model_settings.py'
        
        if _model_settings_file.exists():
            # Load the module directly from file
            spec = importlib.util.spec_from_file_location('agents.model_settings', _model_settings_file)
            if spec and spec.loader:
                _model_settings_module = importlib.util.module_from_spec(spec)
                _model_settings_module.__name__ = 'agents.model_settings'
                _model_settings_module.__package__ = 'agents'
                # Execute the module
                spec.loader.exec_module(_model_settings_module)
                # Register it in sys.modules
                sys.modules['agents.model_settings'] = _model_settings_module
                # Also set it as an attribute on the agents module
                setattr(_agents_proxy, 'model_settings', _model_settings_module)
    except (ImportError, FileNotFoundError, AttributeError):
        pass
    
    # Now import the main vendored module - its internal imports will find 'agents' in sys.modules
    import timestep._vendored.agents as _vendored_agents_module
    
    # Replace the proxy with the real module and update all submodules
    sys.modules['agents'] = _vendored_agents_module
    # Make sure model_settings is still accessible
    if 'agents.model_settings' in sys.modules:
        setattr(_vendored_agents_module, 'model_settings', sys.modules['agents.model_settings'])
    
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

