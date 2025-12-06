"""Compatibility layer for importing vendored or installed agents package."""

import sys

# Try to import from vendored code first (for published packages), 
# then fall back to installed package (for development)
try:
    # Add sys.modules shim so vendored code's absolute imports work
    # This must happen BEFORE importing from _vendored.agents
    import timestep._vendored.agents as _vendored_agents_module
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

