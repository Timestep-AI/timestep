import secrets
import string
import json
import time
from typing import Any, AsyncIterator, Optional
from ._vendored_imports import Model, ModelResponse, Usage, ModelSettings, ModelTracing, TResponseInputItem, Handoff, Tool
from openai.types.responses import ResponseCompletedEvent, ResponseOutputItemDoneEvent, ResponseOutputItemAddedEvent, Response, ResponseTextDeltaEvent, ResponseTextDoneEvent
from openai.types.responses.response_usage import ResponseUsage, InputTokensDetails, OutputTokensDetails
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall


def generate_openai_id(prefix: str, length: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return prefix + ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_tool_call_id() -> str:
    return generate_openai_id('call_', 24)


def generate_completion_id() -> str:
    return generate_openai_id('chatcmpl-', 29)


class OllamaModel(Model):
    """Ollama Model implementation that converts Ollama responses to OpenAI format.
    
    This matches the TypeScript ollama_model.ts implementation as closely as possible,
    adapted to Python's Model interface which uses individual parameters instead of a ModelRequest object.
    """

    def __init__(self, model: str, ollama_client: Any):
        self._client = ollama_client
        self._model = model

    def _convert_ollama_to_openai(self, ollama_response: dict[str, Any]) -> dict[str, Any]:
        """Convert Ollama response format to OpenAI format.
        
        Matches TypeScript _convertOllamaToOpenai implementation.
        """
        ollama_message = ollama_response.get('message', {})

        message: dict[str, Any] = {
            'role': ollama_message.get('role'),
            'content': ollama_message.get('content'),
        }

        tool_calls = ollama_message.get('tool_calls', [])
        if tool_calls:
            message['tool_calls'] = [
                {
                    'id': (
                        tool_call.get('id')
                        if tool_call.get('id') and isinstance(tool_call.get('id'), str) and tool_call.get('id').startswith('call_')
                        else generate_tool_call_id()
                    ),
                    'type': 'function',
                    'function': {
                        'name': tool_call['function']['name'],
                        'arguments': json.dumps(tool_call['function']['arguments']),
                    },
                }
                for tool_call in tool_calls
            ]

        choice = {
            'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop',
            'index': 0,
            'message': message,
        }

        eval_count = ollama_response.get('eval_count', 0)
        prompt_eval_count = ollama_response.get('prompt_eval_count', 0)
        total_tokens = eval_count + prompt_eval_count

        usage = {
            'completion_tokens': eval_count,
            'prompt_tokens': prompt_eval_count,
            'total_tokens': total_tokens,
        }

        result = {
            'id': generate_completion_id(),
            'choices': [choice],
            'created': int(time.time()),
            'model': self._model,
            'object': 'chat.completion',
            'usage': usage,
        }

        return result

    def _convert_handoff_tool(self, handoff: Handoff) -> dict[str, Any]:
        """Convert handoff to tool format.
        
        Matches TypeScript convertHandoffTool implementation.
        """
        # Access handoff attributes - Python Handoff has different structure
        tool_name = getattr(handoff, 'tool_name', getattr(handoff, 'toolName', ''))
        tool_description = getattr(handoff, 'tool_description', getattr(handoff, 'toolDescription', ''))
        input_json_schema = getattr(handoff, 'input_json_schema', getattr(handoff, 'inputJsonSchema', {}))
        
        return {
            'type': 'function',
            'function': {
                'name': tool_name,
                'description': tool_description or '',
                'parameters': input_json_schema,
            },
        }

    async def _fetch_response(
        self,
        system_instructions: Optional[str],
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        handoffs: list[Handoff],
        stream: bool,
    ) -> Any:
        """Fetch response from Ollama, converting format as needed.
        
        Matches TypeScript #fetchResponse implementation, adapted for Python Model interface.
        """
        converted_messages: list[dict[str, Any]] = []

        # Convert input to messages format
        if isinstance(input, str):
            converted_messages = [{'role': 'user', 'content': input}]
        else:
            for item in input:
                if isinstance(item, dict):
                    if item.get('role') == 'tool':
                        converted_messages.append({
                            'role': 'tool',
                            'content': item.get('content', ''),
                            'tool_call_id': item.get('tool_call_id', ''),
                        })
                    elif item.get('type') == 'function_call':
                        parsed_arguments = item.get('arguments')
                        try:
                            if isinstance(parsed_arguments, str):
                                parsed_arguments = json.loads(parsed_arguments)
                        except (json.JSONDecodeError, TypeError):
                            pass

                        converted_messages.append({
                            'role': 'assistant',
                            'content': '',
                            'tool_calls': [
                                {
                                    'id': item.get('callId') or item.get('call_id'),
                                    'type': 'function',
                                    'function': {
                                        'name': item.get('name'),
                                        'arguments': parsed_arguments,
                                    },
                                }
                            ],
                        })
                    elif item.get('type') == 'function_call_result':
                        output = item.get('output')
                        if isinstance(output, str):
                            content = output
                        elif isinstance(output, dict):
                            content = output.get('text') or output.get('content') or json.dumps(output) or ''
                        else:
                            content = json.dumps(output) if output is not None else ''

                        converted_messages.append({
                            'role': 'tool',
                            'content': content,
                            'tool_call_id': item.get('callId') or item.get('call_id'),
                        })
                    elif item.get('role'):
                        msg: dict[str, Any] = {
                            'role': item.get('role'),
                            'content': item.get('content') or item.get('text', ''),
                        }

                        if item.get('tool_calls'):
                            msg['tool_calls'] = item.get('tool_calls')

                        converted_messages.append(msg)
                    else:
                        converted_messages.append({
                            'role': 'user',
                            'content': item.get('content') or item.get('text', ''),
                        })
                else:
                    # Handle non-dict items
                    converted_messages.append({
                        'role': 'user',
                        'content': str(item),
                    })

        # Add system instructions
        if system_instructions:
            converted_messages.insert(0, {
                'content': system_instructions,
                'role': 'system',
            })

        # Convert messages to Ollama format with content extraction
        ollama_messages = []
        for msg in converted_messages:
            content = ''
            msg_content = msg.get('content', '')
            if isinstance(msg_content, str):
                content = msg_content
            elif isinstance(msg_content, list):
                for part in msg_content:
                    if isinstance(part, dict):
                        if part.get('type') == 'input_text' and part.get('text'):
                            content += part.get('text', '')
                        elif part.get('text'):
                            content += part.get('text', '')
                    elif isinstance(part, str):
                        content += part
            elif isinstance(msg_content, dict) and msg_content.get('text'):
                content = msg_content.get('text', '')

            ollama_msg: dict[str, Any] = {
                'role': msg.get('role'),
                'content': content,
            }

            if msg.get('role') == 'tool' and msg.get('tool_call_id'):
                ollama_msg['tool_call_id'] = msg.get('tool_call_id')

            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                ollama_msg['tool_calls'] = []
                for tool_call in msg.get('tool_calls', []):
                    result = dict(tool_call)
                    if result.get('function') and result['function'].get('arguments'):
                        if isinstance(result['function']['arguments'], str):
                            try:
                                result['function']['arguments'] = json.loads(result['function']['arguments'])
                            except json.JSONDecodeError:
                                result['function']['arguments'] = {}
                    ollama_msg['tool_calls'].append(result)

            ollama_messages.append(ollama_msg)

        # Convert tools format
        ollama_tools = []
        if tools:
            for tool in tools:
                # Handle FunctionTool objects (from function_tool decorator)
                if hasattr(tool, 'name') and hasattr(tool, 'params_json_schema'):
                    ollama_tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool.name,
                            'description': tool.description or '',
                            'parameters': tool.params_json_schema or {},
                        },
                    })
                # Handle Tool objects with type='function'
                elif hasattr(tool, 'type') and tool.type == 'function':
                    ollama_tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool.name,
                            'description': tool.description or '',
                            'parameters': tool.parameters or {},
                        },
                    })
                # Handle dict-based tools
                elif isinstance(tool, dict) and tool.get('type') == 'function':
                    ollama_tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool.get('name'),
                            'description': tool.get('description', ''),
                            'parameters': tool.get('parameters', {}),
                        },
                    })

        # Handle handoffs - convert to tools
        if handoffs:
            for handoff in handoffs:
                try:
                    handoff_tool = self._convert_handoff_tool(handoff)
                    if handoff_tool:
                        ollama_tools.append(handoff_tool)
                except Exception:
                    # Silently skip handoffs that can't be converted
                    pass

        # Prepare chat options
        chat_options: dict[str, Any] = {
            'model': self._model,
            'messages': ollama_messages,
            'stream': stream,
        }

        # Apply model settings - match TypeScript implementation
        if hasattr(model_settings, 'temperature') and model_settings.temperature is not None:
            chat_options['temperature'] = model_settings.temperature

        # Handle reasoning settings - map reasoning.effort to think
        # OpenAI Agents SDK: reasoning: { effort: 'minimal' | 'low' | 'medium' | 'high' }
        # Ollama: think: boolean | 'low' | 'medium' | 'high'
        if hasattr(model_settings, 'reasoning') and model_settings.reasoning is not None:
            reasoning = model_settings.reasoning
            if isinstance(reasoning, dict) and reasoning is not None:
                effort = reasoning.get('effort')
                if effort == 'minimal':
                    chat_options['think'] = 'low'  # Map minimal to low
                elif effort in ['low', 'medium', 'high']:
                    chat_options['think'] = effort
                elif effort is None:
                    chat_options['think'] = False  # Disable thinking
            elif reasoning is False:
                chat_options['think'] = False  # Disable thinking mode for consistent responses

        # Map standard model settings to Ollama options
        if 'options' not in chat_options:
            chat_options['options'] = {}
        if hasattr(model_settings, 'temperature') and model_settings.temperature is not None:
            chat_options['options']['temperature'] = model_settings.temperature
        if hasattr(model_settings, 'top_p') and model_settings.top_p is not None:
            chat_options['options']['top_p'] = model_settings.top_p
        if hasattr(model_settings, 'frequency_penalty') and model_settings.frequency_penalty is not None:
            chat_options['options']['frequency_penalty'] = model_settings.frequency_penalty
        if hasattr(model_settings, 'presence_penalty') and model_settings.presence_penalty is not None:
            chat_options['options']['presence_penalty'] = model_settings.presence_penalty

        if ollama_tools:
            chat_options['tools'] = ollama_tools

        # Call Ollama
        response_data = await self._client.chat(**chat_options)

        if stream:
            return response_data

        ret = self._convert_ollama_to_openai(response_data)
        return ret

    def _to_response_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        """Convert usage format.
        
        Matches TypeScript toResponseUsage implementation.
        """
        return {
            'requests': 1,
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'input_tokens_details': {
                'cached_tokens': usage.get('prompt_tokens_details', {}).get('cached_tokens', 0),
            },
            'output_tokens_details': {
                'reasoning_tokens': usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0),
            },
        }

    async def get_response(
        self,
        system_instructions: Optional[str],
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: Optional[Any],
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        prompt: Optional[Any] = None,
    ) -> ModelResponse:
        """Get a response from the Ollama model.
        
        Matches TypeScript getResponse implementation, adapted for Python Model interface.
        """
        # Fetch response (span handling would need to be implemented based on Python agents library API)
        response = await self._fetch_response(
            system_instructions, input, model_settings, tools, handoffs, False
        )

        # Build output - match TypeScript output building logic
        output = []
        if response.get('choices') and response['choices']:
            message = response['choices'][0].get('message', {})

            if (
                message.get('content') is not None
                and message.get('content') != ''
                and not (message.get('tool_calls') and message.get('content') == '')
            ):
                content = message.get('content', '')
                rest = {k: v for k, v in message.items() if k != 'content'}
                output.append({
                    'id': response.get('id'),
                    'type': 'message',
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'output_text',
                            'text': content or '',
                            'provider_data': rest,
                        }
                    ],
                    'status': 'completed',
                })
            elif message.get('refusal'):
                refusal = message.get('refusal')
                rest = {k: v for k, v in message.items() if k != 'refusal'}
                output.append({
                    'id': response.get('id'),
                    'type': 'message',
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'refusal',
                            'refusal': refusal or '',
                            'provider_data': rest,
                        }
                    ],
                    'status': 'completed',
                })
            elif message.get('tool_calls'):
                for tool_call in message.get('tool_calls', []):
                    if tool_call.get('type') == 'function':
                        call_id = tool_call.get('id')
                        remaining_tool_call_data = {k: v for k, v in tool_call.items() if k != 'id'}
                        args = tool_call['function'].get('arguments', '')
                        name = tool_call['function'].get('name')
                        remaining_function_data = {k: v for k, v in tool_call['function'].items() if k not in ['arguments', 'name']}
                        output.append({
                            'id': response.get('id'),
                            'type': 'function_call',
                            'arguments': args,
                            'name': name,
                            'call_id': call_id,
                            'status': 'completed',
                            'provider_data': {
                                **remaining_tool_call_data,
                                **remaining_function_data,
                            },
                        })

        # Build ModelResponse
        usage_obj = Usage(**self._to_response_usage(response.get('usage', {}))) if response.get('usage') else Usage()

        model_response = ModelResponse(
            usage=usage_obj,
            output=output,
            response_id=response.get('id'),
        )

        return model_response

    async def stream_response(
        self,
        system_instructions: Optional[str],
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: Optional[Any],
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        prompt: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        """Stream a response from the Ollama model.
        
        Matches TypeScript getStreamedResponse implementation, adapted for Python Model interface.
        """
        # Note: Span handling would need to be implemented based on Python agents library API
        try:
            stream = await self._fetch_response(
                system_instructions, input, model_settings, tools, handoffs, True
            )

            async for event in self._convert_ollama_stream_to_responses(stream, None, tracing.enabled if hasattr(tracing, 'enabled') else False):
                yield event
        except Exception as error:
            # Error handling would need span support
            raise error

    async def _convert_ollama_stream_to_responses(
        self, stream: Any, span: Optional[Any], tracing_enabled: bool
    ) -> AsyncIterator[Any]:
        """Convert Ollama stream to response events.
        
        Matches TypeScript convertOllamaStreamToResponses implementation.
        Returns Response API event objects.
        """
        usage: Optional[dict[str, Any]] = None
        accumulated_text = ''
        response_id = generate_completion_id()
        # Generate a temporary item_id for streaming events
        item_id = generate_completion_id()

        async for chunk in stream:
            if chunk.get('eval_count') or chunk.get('prompt_eval_count'):
                usage = {
                    'prompt_tokens': chunk.get('prompt_eval_count', 0),
                    'completion_tokens': chunk.get('eval_count', 0),
                    'total_tokens': (chunk.get('prompt_eval_count', 0) or 0) + (chunk.get('eval_count', 0) or 0),
                }

            if chunk.get('message') and chunk['message'].get('content'):
                # Yield ResponseTextDeltaEvent for streaming text
                yield ResponseTextDeltaEvent(
                    item_id=item_id,
                    content_index=0,
                    output_index=0,
                    delta=chunk['message']['content'],
                    sequence_number=1,
                    type='response.output_text.delta',
                    logprobs=[],
                )
                accumulated_text += chunk['message']['content']

            if chunk.get('message') and chunk['message'].get('tool_calls'):
                for tool_call in chunk['message'].get('tool_calls', []):
                    if tool_call.get('function'):
                        call_id = (
                            tool_call.get('id')
                            if tool_call.get('id') and isinstance(tool_call.get('id'), str) and tool_call.get('id').startswith('call_')
                            else generate_tool_call_id()
                        )

                        # Create output with tool call - use proper ResponseFunctionToolCall type
                        tool_call_output_item = ResponseFunctionToolCall(
                            id=response_id,
                            type='function_call',
                            call_id=call_id,
                            name=tool_call['function'].get('name'),
                            arguments=json.dumps(tool_call['function'].get('arguments', {})),
                        )
                        tool_call_output = [tool_call_output_item]

                        response_usage_for_tool = None
                        if usage:
                            response_usage_for_tool = ResponseUsage(
                                input_tokens=usage.get('prompt_tokens', 0),
                                output_tokens=usage.get('completion_tokens', 0),
                                total_tokens=usage.get('total_tokens', 0),
                                input_tokens_details=InputTokensDetails(cached_tokens=0),
                                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                            )
                        
                        response_for_tool = Response(
                            id=response_id,
                            created_at=int(time.time()),
                            model='ollama',
                            object='response',
                            output=tool_call_output,
                            parallel_tool_calls=False,
                            tool_choice='none',  # Use 'none' instead of None
                            tools=[],
                            usage=response_usage_for_tool,
                        )

                        if span and tracing_enabled:
                            if hasattr(span, 'span_data'):
                                span.span_data['output'] = tool_call_output

                        yield ResponseCompletedEvent(
                            response=response_for_tool,
                            sequence_number=1,
                            type='response.completed',
                        )
                        return

            if chunk.get('done'):
                outputs = []

                if accumulated_text:
                    # Yield response.output_text.done event for chatkit
                    yield ResponseTextDoneEvent(
                        item_id=item_id,
                        content_index=0,
                        output_index=0,
                        text=accumulated_text,
                        sequence_number=1,
                        type='response.output_text.done',
                        logprobs=[],
                    )

                    # Create proper ResponseOutputMessage with ResponseOutputText content
                    output_text = ResponseOutputText(
                        type='output_text',
                        text=accumulated_text,
                        annotations=[],
                        logprobs=None,
                    )
                    output_message = ResponseOutputMessage(
                        id=item_id,  # Use item_id for message id to match deltas
                        type='message',
                        role='assistant',
                        status='completed',
                        content=[output_text],
                    )
                    outputs.append(output_message)

                    # Yield response.output_item.added event for chatkit (needed before done)
                    yield ResponseOutputItemAddedEvent(
                        item=output_message,
                        output_index=0,
                        sequence_number=1,
                        type='response.output_item.added',
                    )

                    # Yield response.output_item.done event for chatkit
                    yield ResponseOutputItemDoneEvent(
                        item=output_message,
                        output_index=0,
                        sequence_number=1,
                        type='response.output_item.done',
                    )

                if span and tracing_enabled:
                    if hasattr(span, 'span_data'):
                        span.span_data['output'] = outputs

                # Yield ResponseCompletedEvent for agents library to extract final_response
                response_usage = None
                if usage:
                    response_usage = ResponseUsage(
                        input_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('completion_tokens', 0),
                        total_tokens=usage.get('total_tokens', 0),
                        input_tokens_details=InputTokensDetails(cached_tokens=0),
                        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                    )
                
                response = Response(
                    id=response_id,
                    created_at=int(time.time()),
                    model='ollama',
                    object='response',
                    output=outputs,
                    parallel_tool_calls=False,
                    tool_choice='none',  # Use 'none' instead of None
                    tools=[],
                    usage=response_usage,
                )
                
                yield ResponseCompletedEvent(
                    response=response,
                    sequence_number=1,
                    type='response.completed',
                )
                break
