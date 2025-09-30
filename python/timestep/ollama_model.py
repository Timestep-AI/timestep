#!/usr/bin/env python3
"""
Ollama model implementation for Agents SDK using Chat Completions API pattern
"""
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncIterable

from ollama import AsyncClient

from agents import Model, ModelResponse
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from agents import ModelSettings, Tool, Handoff, ModelTracing, AgentOutputSchemaBase, TResponseStreamEvent

def generate_tool_call_id():
    return f"call_{uuid.uuid4().hex[:24]}"

def generate_completion_id():
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"

class OllamaModel(Model):
    """Ollama model using Chat Completions API pattern"""

    def __init__(self, model: str, ollama_client: AsyncClient = None, api_key: str = None, base_url: str = "https://ollama.com"):
        self._model = model
        if ollama_client:
            self._client = ollama_client
        else:
            self._client = AsyncClient(host=base_url, headers={'Authorization': api_key})

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, *, previous_response_id=None, conversation_id=None, prompt=None) -> ModelResponse:
        """Get non-streaming response"""
        # Convert request to Ollama format
        messages = self._convert_input_to_messages(input, system_instructions)

        # Prepare tools
        ollama_tools = []
        if tools:
            for tool in tools:
                if hasattr(tool, 'name'):
                    ollama_tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool.name,
                            'description': getattr(tool, 'description', ''),
                            'parameters': getattr(tool, 'parameters', {}) or getattr(tool, 'params_json_schema', {}),
                        },
                    })

        # Add handoffs as tools
        if handoffs:
            for handoff in handoffs:
                try:
                    handoff_tool = self._convert_handoff_to_tool(handoff)
                    if handoff_tool:
                        ollama_tools.append(handoff_tool)
                except Exception:
                    pass

        # Prepare chat options
        chat_options = {
            'model': self._model,
            'messages': messages,
            'stream': False,
        }

        if ollama_tools:
            chat_options['tools'] = ollama_tools

        # Make the request
        response = await self._client.chat(**chat_options)

        # Convert response to model format
        return self._convert_ollama_response(response)

    async def stream_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, *, previous_response_id=None, conversation_id=None, prompt=None) -> AsyncIterator['TResponseStreamEvent']:
        """Stream response using Chat Completions API pattern like OpenAI"""

        # Convert request to Ollama format
        messages = self._convert_input_to_messages(input, system_instructions)

        # Prepare tools
        ollama_tools = []
        if tools:
            for tool in tools:
                if hasattr(tool, 'name'):
                    ollama_tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool.name,
                            'description': getattr(tool, 'description', ''),
                            'parameters': getattr(tool, 'parameters', {}) or getattr(tool, 'params_json_schema', {}),
                        },
                    })

        if handoffs:
            for handoff in handoffs:
                try:
                    handoff_tool = self._convert_handoff_to_tool(handoff)
                    if handoff_tool:
                        ollama_tools.append(handoff_tool)
                except Exception:
                    pass

        # Prepare chat options
        chat_options = {
            'model': self._model,
            'messages': messages,
            'stream': True,
        }

        if ollama_tools:
            chat_options['tools'] = ollama_tools

        # Start streaming
        stream = await self._client.chat(**chat_options)

        # Convert Ollama stream to OpenAI-compatible chat completion chunks
        chat_completion_stream = self._convert_ollama_to_chat_completion_stream(stream)

        # Use the standard ChatCmplStreamHandler to process the stream
        from agents.models.chatcmpl_stream_handler import ChatCmplStreamHandler
        from openai.types.responses import Response

        # Create a fake response for the handler
        fake_response = Response(
            id=generate_completion_id(),
            created_at=time.time(),
            model=self._model,
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )

        async for event in ChatCmplStreamHandler.handle_stream(fake_response, chat_completion_stream):
            yield event

    def _convert_ollama_to_chat_completion_stream(self, ollama_stream):
        """Convert Ollama stream to OpenAI chat completion format"""
        async def chat_completion_generator():
            completion_id = generate_completion_id()
            created = int(time.time())
            first_chunk = True

            async for chunk in ollama_stream:
                from openai.types.chat import ChatCompletionChunk
                from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

                # Handle first chunk
                if first_chunk:
                    first_chunk = False
                    delta = ChoiceDelta(role='assistant')
                    choice = Choice(index=0, delta=delta, finish_reason=None)
                    chat_chunk = ChatCompletionChunk(
                        id=completion_id,
                        choices=[choice],
                        created=created,
                        model=self._model,
                        object='chat.completion.chunk'
                    )
                    yield chat_chunk
                    continue

                # Handle message content
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content') and chunk.message.content:
                    delta = ChoiceDelta(content=chunk.message.content)
                    choice = Choice(index=0, delta=delta, finish_reason=None)
                    chat_chunk = ChatCompletionChunk(
                        id=completion_id,
                        choices=[choice],
                        created=created,
                        model=self._model,
                        object='chat.completion.chunk'
                    )
                    yield chat_chunk

                # Handle tool calls
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'tool_calls') and chunk.message.tool_calls:
                    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

                    tool_calls_delta = []
                    for i, tool_call in enumerate(chunk.message.tool_calls):
                        if hasattr(tool_call, 'function'):
                            call_id = getattr(tool_call, 'id', '') or generate_tool_call_id()
                            tool_call_delta = ChoiceDeltaToolCall(
                                index=i,
                                id=call_id,
                                type='function',
                                function=ChoiceDeltaToolCallFunction(
                                    name=tool_call.function.name,
                                    arguments=json.dumps(tool_call.function.arguments)
                                )
                            )
                            tool_calls_delta.append(tool_call_delta)

                    if tool_calls_delta:
                        delta = ChoiceDelta(tool_calls=tool_calls_delta)
                        choice = Choice(index=0, delta=delta, finish_reason=None)
                        chat_chunk = ChatCompletionChunk(
                            id=completion_id,
                            choices=[choice],
                            created=created,
                            model=self._model,
                            object='chat.completion.chunk'
                        )
                        yield chat_chunk

                # Handle completion
                if hasattr(chunk, 'done') and chunk.done:
                    # Add usage information if available
                    usage = None
                    if hasattr(chunk, 'eval_count') or hasattr(chunk, 'prompt_eval_count'):
                        from openai.types.completion_usage import CompletionUsage
                        prompt_tokens = getattr(chunk, 'prompt_eval_count', 0) or 0
                        completion_tokens = getattr(chunk, 'eval_count', 0) or 0
                        usage = CompletionUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens
                        )

                    # Final chunk with finish reason
                    finish_reason = 'tool_calls' if (hasattr(chunk, 'message') and
                                                    hasattr(chunk.message, 'tool_calls') and
                                                    chunk.message.tool_calls) else 'stop'

                    delta = ChoiceDelta()
                    choice = Choice(index=0, delta=delta, finish_reason=finish_reason)
                    chat_chunk = ChatCompletionChunk(
                        id=completion_id,
                        choices=[choice],
                        created=created,
                        model=self._model,
                        object='chat.completion.chunk',
                        usage=usage
                    )
                    yield chat_chunk
                    break

        return chat_completion_generator()

    def _convert_input_to_messages(self, input_data: Any, system_instructions: Optional[str] = None) -> List[Dict[str, Any]]:
        """Convert input to Ollama message format"""
        messages = []

        if system_instructions:
            messages.append({'role': 'system', 'content': system_instructions})

        if isinstance(input_data, str):
            messages.append({'role': 'user', 'content': input_data})
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, dict):
                    item_type = item.get('type', '')
                    if item_type == 'function_call_output':
                        # Tool result
                        content = item.get('output', '')
                        if isinstance(content, dict) and 'text' in content:
                            content = content['text']
                        elif isinstance(content, dict) and 'content' in content:
                            content = content['content']
                        messages.append({
                            'role': 'tool',
                            'content': str(content),
                            'tool_call_id': item.get('call_id', ''),
                        })
                    elif item_type == 'function_call':
                        # Previous function call
                        arguments = item.get('arguments', '{}')
                        if isinstance(arguments, str):
                            try:
                                parsed_args = json.loads(arguments)
                            except:
                                parsed_args = {}
                        else:
                            parsed_args = arguments

                        messages.append({
                            'role': 'assistant',
                            'content': '',
                            'tool_calls': [{
                                'id': item.get('call_id', ''),
                                'type': 'function',
                                'function': {
                                    'name': item.get('name', ''),
                                    'arguments': parsed_args,
                                },
                            }],
                        })
                    else:
                        # Regular message
                        messages.append({
                            'role': item.get('role', 'user'),
                            'content': item.get('content', '') or item.get('text', '')
                        })
                elif hasattr(item, 'role'):
                    messages.append({
                        'role': item.role,
                        'content': getattr(item, 'content', '') or getattr(item, 'text', '')
                    })

        return messages

    def _convert_handoff_to_tool(self, handoff: Any) -> Optional[Dict[str, Any]]:
        """Convert handoff to tool format"""
        # Check for different possible attributes for the agent name
        agent_name = None
        if hasattr(handoff, 'agent_name'):
            agent_name = handoff.agent_name
        elif hasattr(handoff, 'name'):
            agent_name = handoff.name
        elif hasattr(handoff, 'tool_name'):
            # Use the existing tool_name if available
            return {
                'type': 'function',
                'function': {
                    'name': handoff.tool_name,
                    'description': getattr(handoff, 'tool_description', f'Handoff to {handoff.tool_name}'),
                    'parameters': {
                        'type': 'object',
                        'properties': {},
                        'required': [],
                        'additionalProperties': False
                    },
                },
            }

        if agent_name:
            return {
                'type': 'function',
                'function': {
                    'name': f'transfer_to_{agent_name.replace(" ", "_").lower()}',
                    'description': getattr(handoff, 'handoff_description', f'Transfer to {agent_name}'),
                    'parameters': {
                        'type': 'object',
                        'properties': {},
                        'required': [],
                        'additionalProperties': False
                    },
                },
            }
        return None

    def _convert_ollama_response(self, ollama_response: Any) -> ModelResponse:
        """Convert Ollama response to ModelResponse"""
        from agents.types.model import (
            OutputModelItem, FunctionCallItem, TextOutputItem, Usage
        )

        output_items: List[OutputModelItem] = []

        if hasattr(ollama_response, 'message'):
            message = ollama_response.message

            # Handle tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function'):
                        call_id = getattr(tool_call, 'id', '') or generate_tool_call_id()
                        args_str = json.dumps(tool_call.function.arguments)

                        output_items.append(FunctionCallItem(
                            arguments=args_str,
                            call_id=call_id,
                            name=tool_call.function.name,
                            type='function_call'
                        ))

            # Handle text content
            if hasattr(message, 'content') and message.content:
                output_items.append(TextOutputItem(
                    content=message.content,
                    type='text_output'
                ))

        # Create usage info
        usage = None
        if hasattr(ollama_response, 'prompt_eval_count') or hasattr(ollama_response, 'eval_count'):
            from agents.types.model import Usage, UsageData
            usage = Usage(
                requests=1,
                input_tokens=getattr(ollama_response, 'prompt_eval_count', 0) or 0,
                output_tokens=getattr(ollama_response, 'eval_count', 0) or 0,
                total_tokens=(getattr(ollama_response, 'prompt_eval_count', 0) or 0) + (getattr(ollama_response, 'eval_count', 0) or 0)
            )

        return ModelResponse(
            output=output_items,
            usage=usage,
            response_id=generate_completion_id()
        )