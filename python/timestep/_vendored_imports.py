"""Compatibility layer for importing vendored or installed agents package."""

import sys
import types
import importlib
import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path

# Custom import finder to redirect 'agents.*' imports to 'timestep._vendored.agents.*'
class _VendoredAgentsFinder:
    """MetaPathFinder that redirects 'agents.*' imports to vendored location."""
    
    def find_spec(self, name, path, target=None):
        if name == 'agents':
            # For the main 'agents' module, we'll handle it after import
            # Return None to let normal import handle it
            return None
        elif name.startswith('agents.'):
            # For submodules like 'agents.model_settings', redirect to vendored location
            # But import the submodule directly, not through the main package
            submodule_name = name[len('agents.'):]
            vendored_name = f'timestep._vendored.agents.{submodule_name}'
            try:
                spec = importlib.util.find_spec(vendored_name)
                if spec:
                    # Create a new spec with the original name so imports work correctly
                    return ModuleSpec(
                        name=name,
                        loader=spec.loader,
                        origin=spec.origin,
                        loader_state=spec.loader_state,
                        is_package=spec.is_package,
                        submodule_search_locations=spec.submodule_search_locations
                    )
            except (ImportError, ValueError, AttributeError):
                pass
        return None

# Try to import from vendored code first (for published packages), 
# then fall back to installed package (for development)
try:
    # Install the custom finder FIRST, before any imports
    _finder = _VendoredAgentsFinder()
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
    
    # Pre-import model_settings to avoid circular imports during main package initialization
    # This must happen before importing the main agents module
    try:
        import timestep._vendored.agents.model_settings as _model_settings_module
        sys.modules['agents.model_settings'] = _model_settings_module
        # Create a temporary 'agents' module to hold it
        _temp_agents = types.ModuleType('agents')
        _temp_agents.model_settings = _model_settings_module
        sys.modules['agents'] = _temp_agents
    except ImportError:
        pass
    
    # Now import the main vendored module - the finder will redirect 'agents.*' imports
    import timestep._vendored.agents as _vendored_agents_module
    
    # Register the vendored module as 'agents' in sys.modules
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
    from ._vendored.agents.guardrail import InputGuardrail, OutputGuardrail
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
    from agents.guardrail import InputGuardrail, OutputGuardrail
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
    'InputGuardrail', 'OutputGuardrail',
    'AgentsException', 'MaxTurnsExceeded', 'ModelBehaviorError', 'UserError',
    'InputGuardrailTripwireTriggered', 'OutputGuardrailTripwireTriggered',
    'SessionABC',
]

