// Ollama will be dynamically imported when needed
import {
  Model,
  Usage,
  withGenerationSpan,
  resetCurrentSpan,
  createGenerationSpan,
  setCurrentSpan,
} from '@openai/agents-core';
import type { ModelRequest, ModelResponse, ResponseStreamEvent } from '@openai/agents-core';
import { protocol } from '@openai/agents-core';
import { Span } from '@openai/agents-core';

function generateOpenAIId(prefix: string, length: number): string {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const bytes = new Uint8Array(length);
  (globalThis as any).crypto.getRandomValues(bytes);
  let out = '';
  for (let i = 0; i < length; i++) out += alphabet[bytes[i] % alphabet.length];
  return `${prefix}${out}`;
}

function generateToolCallId(): string {
  return generateOpenAIId('call_', 24);
}

function generateCompletionId(): string {
  return generateOpenAIId('chatcmpl-', 29);
}

function generateMessageId(): string {
  return generateOpenAIId('msg_', 27);
}

/**
 * Recursively remove tool_calls from an object to prevent them from being
 * included in providerData when saving to Conversations API.
 * Also removes any objects with type: 'function' that might be extracted as separate items.
 */
function removeToolCallsRecursively(obj: any): any {
  if (!obj || typeof obj !== 'object') {
    return obj;
  }
  
  // If it's an array, filter out tool_calls and objects with type: 'function'
  if (Array.isArray(obj)) {
    return obj
      .filter((item) => {
        // Filter out tool_calls arrays and objects with type: 'function'
        if (item && typeof item === 'object') {
          return item.type !== 'function' && !('tool_calls' in item && Array.isArray(item.tool_calls));
        }
        return true;
      })
      .map((item) => removeToolCallsRecursively(item));
  }
  
  // If it's an object with type: 'function', return undefined to exclude it
  if (obj.type === 'function') {
    return undefined;
  }
  
  const result: any = {};
  for (const [key, value] of Object.entries(obj)) {
    if (key === 'tool_calls' || key === 'toolCalls') {
      continue; // Skip tool_calls
    }
    const cleaned = removeToolCallsRecursively(value);
    // Only include the key if the cleaned value is not undefined
    if (cleaned !== undefined) {
      result[key] = cleaned;
    }
  }
  return result;
}

export class OllamaModel implements Model {
  #client: any; // Will be dynamically imported
  #model: string;

  constructor(model: string, ollama_client: any) {
    this.#client = ollama_client;
    this.#model = model;
  }

  private _convertOllamaToOpenai(ollamaResponse: any): any {
    const ollamaMessage = ollamaResponse['message'];

    const message: any = {
      role: ollamaMessage['role'],
      content: ollamaMessage['content'],
    };

    if (ollamaMessage['tool_calls'] && ollamaMessage['tool_calls'].length > 0) {
      message.tool_calls = ollamaMessage['tool_calls'].map((toolCall: any) => {
        const id =
          toolCall.id && typeof toolCall.id === 'string' && toolCall.id.startsWith('call_')
            ? toolCall.id
            : generateToolCallId();

        return {
          id: id,
          type: 'function',
          function: {
            name: toolCall.function.name,
            arguments: JSON.stringify(toolCall.function.arguments),
          },
        };
      });
    }

    const choice = {
      finish_reason: message.tool_calls ? 'tool_calls' : 'stop',
      index: 0,
      message: message,
    };

    const evalCount = ollamaResponse['eval_count'] || 0;
    const promptEvalCount = ollamaResponse['prompt_eval_count'] || 0;
    const totalTokens = evalCount + promptEvalCount;

    const usage = {
      completion_tokens: evalCount,
      prompt_tokens: promptEvalCount,
      total_tokens: totalTokens,
    };

    const result = {
      id: generateCompletionId(),
      choices: [choice],
      created: Math.floor(Date.now() / 1000),
      model: this.#model,
      object: 'chat.completion',
      usage: usage,
    };

    return result;
  }

  private convertHandoffTool(handoff: any) {
    return {
      type: 'function',
      function: {
        name: handoff.toolName,
        description: handoff.toolDescription || '',
        parameters: handoff.inputJsonSchema,
      },
    };
  }

  async #fetchResponse(
    request: ModelRequest,
    span: Span<any> | undefined,
    stream: true
  ): Promise<any>;
  async #fetchResponse(
    request: ModelRequest,
    span: Span<any> | undefined,
    stream: false
  ): Promise<any>;
  async #fetchResponse(
    request: ModelRequest,
    span: Span<any> | undefined,
    stream: boolean
  ): Promise<any> {
    let convertedMessages: any[] = [];

    if (typeof request.input === 'string') {
      convertedMessages = [{ role: 'user', content: request.input }];
    } else {
      convertedMessages = request.input
        .map((item: any) => {
          if (item.role === 'tool') {
            return {
              role: 'tool',
              content: item.content || '',
              tool_call_id: item.tool_call_id || '',
            };
          } else if (item.type === 'function_call') {
            let parsedArguments;
            try {
              parsedArguments = JSON.parse(item.arguments);
            } catch (e) {
              parsedArguments = item.arguments;
            }

            return {
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  id: item.callId,
                  type: 'function',
                  function: {
                    name: item.name,
                    arguments: parsedArguments,
                  },
                },
              ],
            };
          } else if (item.type === 'function_call_result') {
            let content = '';
            if (typeof item.output === 'string') {
              content = item.output;
            } else if (item.output?.text) {
              content = item.output.text;
            } else if (item.output?.content) {
              content = item.output.content;
            } else {
              content = JSON.stringify(item.output) || '';
            }

            return {
              role: 'tool',
              content: content,
              tool_call_id: item.callId,
            };
          } else if (item.role) {
            const msg: any = {
              role: item.role,
              content: item.content || item.text || '',
            };

            if (item.tool_calls) {
              msg.tool_calls = item.tool_calls;
            }

            return msg;
          } else {
            return {
              role: 'user',
              content: item.content || item.text || '',
            };
          }
        })
        .filter((msg) => msg !== null);
    }

    if (request.systemInstructions) {
      convertedMessages.unshift({
        content: request.systemInstructions,
        role: 'system',
      });
    }

    if (span && request.tracing === true) {
      span.spanData.input = convertedMessages;
    }

    /**
     * Extract and validate base64 string from a data URL.
     * The Ollama client's encodeImage function returns strings as-is, so we pass base64 directly.
     * We validate the base64 is actually valid base64 (similar to Python client's Image class).
     */
    function extractBase64FromImage(image: string): string {
      let base64: string;
      // If it's a data URL (data:image/...;base64,<base64>), extract the base64 part
      if (image.startsWith('data:')) {
        const commaIndex = image.indexOf(',');
        if (commaIndex !== -1) {
          base64 = image.substring(commaIndex + 1);
        } else {
          base64 = image;
        }
      } else {
        // If it's already base64 (no data: prefix), use as-is
        base64 = image;
      }
      
      // Validate that it's actually valid base64 (similar to Python client's Image.serialize_model)
      // This helps catch invalid data before sending to Ollama Cloud
      try {
        // Try to decode to validate it's base64
        Buffer.from(base64, 'base64');
        return base64;
      } catch (e) {
        throw new Error(`Invalid base64 image data: ${e}`);
      }
    }

    const ollamaMessages = [];
    for (const msg of convertedMessages) {
      let content = '';
      const images: string[] = [];
      
      if (typeof msg['content'] === 'string') {
        content = msg['content'];
      } else if (Array.isArray(msg['content'])) {
        for (const part of msg['content']) {
          if (part.type === 'input_text' && part.text) {
            content += part.text;
          } else if (part.type === 'input_image' && part.image) {
            // Extract base64 from data URL - pass base64 string directly
            // The Ollama client's encodeImage returns strings as-is
            const imageStr = typeof part.image === 'string' 
              ? part.image 
              : (part.image as any).id 
                ? undefined // File ID references not supported by Ollama
                : undefined;
            
            if (imageStr) {
              images.push(extractBase64FromImage(imageStr));
            }
          } else if (typeof part === 'string') {
            content += part;
          } else if (part.text) {
            content += part.text;
          }
        }
      } else if (msg['content'] && typeof msg['content'] === 'object' && msg['content'].text) {
        content = msg['content'].text;
      }

      const ollamaMsg: any = {
        role: msg['role'],
        content: content,
      };

      // Add images array if there are any images
      // Pass base64 strings directly - the Ollama client's encodeImage returns strings as-is
      if (images.length > 0) {
        ollamaMsg.images = images;
      }

      if (msg['role'] === 'tool' && msg['tool_call_id']) {
        ollamaMsg['tool_call_id'] = msg['tool_call_id'];
      }

      if (msg['role'] === 'assistant' && msg['tool_calls']) {
        ollamaMsg['tool_calls'] = msg['tool_calls'].map((toolCall: any) => {
          const result = { ...toolCall };
          if (result.function && result.function.arguments) {
            if (typeof result.function.arguments === 'string') {
              try {
                result.function.arguments = JSON.parse(result.function.arguments);
              } catch (error) {
                result.function.arguments = {};
              }
            }
          }
          return result;
        });
      }

      ollamaMessages.push(ollamaMsg);
    }

    const ollamaTools =
      request.tools
        ?.map((tool) => {
          // Handle FunctionTool objects (from tool() function in agents-core)
          if ((tool as any).name && (tool as any).paramsJsonSchema) {
            const paramsSchema = (tool as any).paramsJsonSchema;

            return {
              type: 'function',
              function: {
                name: (tool as any).name,
                description: (tool as any).description || '',
                parameters: paramsSchema || {},
              },
            };
          }
          // Handle Tool objects with type='function'
          if (tool.type === 'function') {
            return {
              type: 'function',
              function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
              },
            };
          }
          return null;
        })
        .filter((tool) => tool !== null) || [];

    if ((request as any).handoffs && Array.isArray((request as any).handoffs)) {
      for (const handoff of (request as any).handoffs) {
        try {
          const handoffTool = this.convertHandoffTool(handoff);
          if (handoffTool) {
            ollamaTools.push(handoffTool);
          }
        } catch (e) {
          // Silently skip handoffs that can't be converted
        }
      }
    }

    const chatOptions: any = {
      model: this.#model,
      messages: ollamaMessages,
      stream: stream as any,
    };

    // Set temperature directly on the request (not in options)
    if (request.modelSettings?.temperature !== undefined) {
      chatOptions.temperature = request.modelSettings.temperature;
    }

    // Add model settings if provided
    if (request.modelSettings) {
      // Handle reasoning settings - map reasoning.effort to think
      // OpenAI Agents SDK: reasoning: { effort: 'minimal' | 'low' | 'medium' | 'high' }
      // Ollama: think: boolean | 'low' | 'medium' | 'high'
      if (request.modelSettings.reasoning !== undefined) {
        if (typeof request.modelSettings.reasoning === 'object' && request.modelSettings.reasoning !== null) {
          const effort = request.modelSettings.reasoning.effort;
          if (effort === 'minimal') {
            chatOptions.think = 'low'; // Map minimal to low
          } else if (effort === 'low' || effort === 'medium' || effort === 'high') {
            chatOptions.think = effort;
          } else if (effort === null) {
            chatOptions.think = false; // Disable thinking
          }
        } else if (request.modelSettings.reasoning === false) {
          chatOptions.think = false; // Disable thinking mode for consistent responses
        }
      }

      // Map standard model settings to Ollama options
      if (request.modelSettings.temperature !== undefined) {
        chatOptions.options = chatOptions.options || {};
        chatOptions.options.temperature = request.modelSettings.temperature;
      }
      if (request.modelSettings.topP !== undefined) {
        chatOptions.options = chatOptions.options || {};
        chatOptions.options.top_p = request.modelSettings.topP;
      }
      if (request.modelSettings.frequencyPenalty !== undefined) {
        chatOptions.options = chatOptions.options || {};
        chatOptions.options.frequency_penalty = request.modelSettings.frequencyPenalty;
      }
      if (request.modelSettings.presencePenalty !== undefined) {
        chatOptions.options = chatOptions.options || {};
        chatOptions.options.presence_penalty = request.modelSettings.presencePenalty;
      }
    }

    if (ollamaTools.length > 0) {
      chatOptions.tools = ollamaTools;
    }

    const responseData = await this.#client.chat(chatOptions);

    if (stream) {
      return responseData;
    }

    const ret = this._convertOllamaToOpenai(responseData);

    return ret;
  }

  private toResponseUsage(usage: any) {
    return {
      requests: 1,
      input_tokens: usage.prompt_tokens || 0,
      output_tokens: usage.completion_tokens || 0,
      total_tokens: usage.total_tokens || 0,
      input_tokens_details: {
        cached_tokens: usage.prompt_tokens_details?.cached_tokens || 0,
      },
      output_tokens_details: {
        reasoning_tokens: usage.completion_tokens_details?.reasoning_tokens || 0,
      },
    };
  }

  async getResponse(request: ModelRequest): Promise<ModelResponse> {
    const response = await withGenerationSpan(async (span) => {
      span.spanData.model = this.#model;
      span.spanData.model_config = request.modelSettings
        ? {
            temperature: request.modelSettings.temperature,
            top_p: request.modelSettings.topP,
            frequency_penalty: request.modelSettings.frequencyPenalty,
            presence_penalty: request.modelSettings.presencePenalty,
          }
        : { base_url: 'ollama_client' };
      const response = await this.#fetchResponse(request, span, false);
      if (span && request.tracing === true) {
        span.spanData.output = [response];
      }
      return response;
    });

    const output: protocol.OutputModelItem[] = [];
    if (response.choices && response.choices[0]) {
      const message = response.choices[0].message;

      if (
        message.content !== undefined &&
        message.content !== null &&
        // Azure OpenAI returns empty string instead of null for tool calls, causing parser rejection
        !(message.tool_calls && message.content === '')
      ) {
        // Exclude tool_calls from providerData - they're for Chat Completions API format only,
        // not for Conversations API which expects function_call items instead
        // Also recursively remove tool_calls from nested objects to prevent them from being
        // included via camelOrSnakeToSnakeCase processing
        // Only include 'role' in providerData if it's different from what we're setting
        const { content, tool_calls, role, ...rest } = message;
        const cleanRest = removeToolCallsRecursively(rest);
        // Only set providerData if there's actually something to include (besides role which we already set)
        const providerData = Object.keys(cleanRest).length > 0 ? cleanRest : undefined;
        output.push({
          id: generateMessageId(),
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'output_text',
              text: content || '',
              ...(providerData ? { providerData } : {}),
            },
          ],
          status: 'completed',
        });
      } else if (message.refusal) {
        // Exclude tool_calls from providerData
        const { refusal, tool_calls, role, ...rest } = message;
        const cleanRest = removeToolCallsRecursively(rest);
        // Only set providerData if there's actually something to include (besides role which we already set)
        const providerData = Object.keys(cleanRest).length > 0 ? cleanRest : undefined;
        output.push({
          id: generateMessageId(),
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'refusal',
              refusal: refusal || '',
              ...(providerData ? { providerData } : {}),
            },
          ],
          status: 'completed',
        });
      } else if (message.tool_calls) {
        for (const tool_call of message.tool_calls) {
          if (tool_call.type === 'function') {
            // Exclude 'type', 'id', and 'function' from tool_call, and 'arguments' and 'name' from function
            // to prevent Chat Completions API format fields from being included in providerData
            const { id: callId, type, function: func, ...remainingToolCallData } = tool_call;
            const { arguments: args, name, ...remainingFunctionData } = func;
            const cleanProviderData = removeToolCallsRecursively({
              ...remainingToolCallData,
              ...remainingFunctionData,
            });
            output.push({
              id: generateMessageId(),
              type: 'function_call',
              arguments: args,
              name: name,
              callId: callId,
              status: 'completed',
              ...(Object.keys(cleanProviderData).length > 0 ? { providerData: cleanProviderData } : {}),
            });
          }
        }
      }
    }

    const modelResponse: ModelResponse = {
      usage: response.usage ? new Usage(this.toResponseUsage(response.usage)) : new Usage(),
      output,
      responseId: response.id,
    };

    return modelResponse;
  }

  async *getStreamedResponse(request: ModelRequest): AsyncIterable<ResponseStreamEvent> {
    const span = request.tracing ? createGenerationSpan() : undefined;
    try {
      if (span) {
        span.start();
        setCurrentSpan(span);
      }
      const stream = await this.#fetchResponse(request, span, true);

      yield* this.convertOllamaStreamToResponses(stream, span, request.tracing === true);
    } catch (error) {
      if (span) {
        span.setError({
          message: 'Error streaming response',
          data: {
            error:
              request.tracing === true
                ? String(error)
                : error instanceof Error
                  ? error.name
                  : undefined,
          },
        });
      }
      throw error;
    } finally {
      if (span) {
        span.end();
        resetCurrentSpan();
      }
    }
  }

  private async *convertOllamaStreamToResponses(
    stream: any,
    span?: Span<any>,
    tracingEnabled?: boolean
  ): AsyncIterable<ResponseStreamEvent> {
    let usage: any = undefined;
    let started = false;
    let accumulatedText = '';
    const responseId = generateCompletionId();

    for await (const chunk of stream) {
      if (!started) {
        started = true;
        // Yield response_started event in TypeScript SDK format
        yield {
          type: 'response_started',
          providerData: chunk,
        };
      }

      // Yield model event in TypeScript SDK format
      yield {
        type: 'model',
        event: chunk,
      };

      if (chunk.eval_count || chunk.prompt_eval_count) {
        usage = {
          prompt_tokens: chunk.prompt_eval_count || 0,
          completion_tokens: chunk.eval_count || 0,
          total_tokens: (chunk.prompt_eval_count || 0) + (chunk.eval_count || 0),
        };
      }

      if (chunk.message && chunk.message.content) {
        yield {
          type: 'output_text_delta',
          delta: chunk.message.content,
          providerData: chunk,
        };
        accumulatedText += chunk.message.content;
      }

      if (chunk.message && chunk.message.tool_calls) {
        for (const tool_call of chunk.message.tool_calls) {
          if (tool_call.function) {
            const callId =
              tool_call.id && typeof tool_call.id === 'string' && tool_call.id.startsWith('call_')
                ? tool_call.id
                : generateToolCallId();

            const functionCallEvent = {
              type: 'response_done' as const,
              response: {
                id: responseId,
                usage: {
                  inputTokens: usage?.prompt_tokens ?? 0,
                  outputTokens: usage?.completion_tokens ?? 0,
                  totalTokens: usage?.total_tokens ?? 0,
                  inputTokensDetails: { cached_tokens: 0 },
                  outputTokensDetails: { reasoning_tokens: 0 },
                },
                output: [
                  {
                    id: generateMessageId(),
                    type: 'function_call' as const,
                    arguments: JSON.stringify(tool_call.function.arguments),
                    name: tool_call.function.name,
                    callId: callId,
                    status: 'completed' as const,
                    providerData: tool_call,
                  },
                ],
              },
            };

            if (span && tracingEnabled === true) {
              span.spanData.output = functionCallEvent.response.output;
            }

            yield functionCallEvent;
            return;
          }
        }
      }

      if (chunk.done) {
        const outputs: protocol.OutputModelItem[] = [];

        if (accumulatedText) {
          outputs.push({
            id: generateMessageId(),
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'output_text',
                text: accumulatedText,
              },
            ],
            status: 'completed',
          });
        }

        if (span && tracingEnabled === true) {
          span.spanData.output = outputs;
        }

        // Yield response_done event in TypeScript SDK format
        yield {
          type: 'response_done' as const,
          response: {
            id: responseId,
            usage: {
              inputTokens: usage?.prompt_tokens ?? 0,
              outputTokens: usage?.completion_tokens ?? 0,
              totalTokens: usage?.total_tokens ?? 0,
              inputTokensDetails: {
                cached_tokens: 0,
              },
              outputTokensDetails: {
                reasoning_tokens: 0,
              },
            },
            output: outputs,
          },
        };
        break;
      }
    }
  }
}
