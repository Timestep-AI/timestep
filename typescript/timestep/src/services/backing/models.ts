/**
 * Custom model implementations for A2A agent library.
 */
// @ts-nocheck

import {Ollama} from 'ollama';
import {
	Model,
	Usage,
	withGenerationSpan,
	resetCurrentSpan,
	createGenerationSpan,
	setCurrentSpan,
} from '@openai/agents-core';
import type {
	ModelRequest,
	ModelResponse,
	ResponseStreamEvent,
	SerializedOutputType,
} from '@openai/agents-core';
import type {Stream} from 'openai/streaming';
import {protocol} from '@openai/agents-core';
import {Span, GenerationSpanData} from '@openai/agents-core';

function generateOpenAIId(prefix: string, length: number): string {
	const alphabet =
		'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
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

// Note: TypeScript SDK has different interfaces than Python, so we need to adapt
// This implementation maintains the same logic as Python but uses TypeScript SDK patterns

export class OllamaModel implements Model {
	#client: Ollama;
	#model: string;

	constructor(client: Ollama, model: string) {
		this.#client = client;
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
					toolCall.id &&
					typeof toolCall.id === 'string' &&
					toolCall.id.startsWith('call_')
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
		span: Span<GenerationSpanData> | undefined,
		stream: true,
	): Promise<Stream<any>>;
	async #fetchResponse(
		request: ModelRequest,
		span: Span<GenerationSpanData> | undefined,
		stream: false,
	): Promise<any>;
	async #fetchResponse(
		request: ModelRequest,
		span: Span<GenerationSpanData> | undefined,
		stream: boolean,
	): Promise<Stream<any> | any> {
		let convertedMessages: any[] = [];

		if (typeof request.input === 'string') {
			convertedMessages = [{role: 'user', content: request.input}];
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
				.filter(msg => msg !== null);
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

		const ollamaMessages = [];
		for (const msg of convertedMessages) {
			let content = '';
			if (typeof msg['content'] === 'string') {
				content = msg['content'];
			} else if (Array.isArray(msg['content'])) {
				for (const part of msg['content']) {
					if (part.type === 'input_text' && part.text) {
						content += part.text;
					} else if (typeof part === 'string') {
						content += part;
					} else if (part.text) {
						content += part.text;
					}
				}
			} else if (
				msg['content'] &&
				typeof msg['content'] === 'object' &&
				msg['content'].text
			) {
				content = msg['content'].text;
			}

			const ollamaMsg: any = {
				role: msg['role'],
				content: content,
			};

			if (msg['role'] === 'tool' && msg['tool_call_id']) {
				ollamaMsg['tool_call_id'] = msg['tool_call_id'];
			}

			if (msg['role'] === 'assistant' && msg['tool_calls']) {
				ollamaMsg['tool_calls'] = msg['tool_calls'].map((toolCall: any) => {
					const result = {...toolCall};
					if (result.function && result.function.arguments) {
						if (typeof result.function.arguments === 'string') {
							try {
								result.function.arguments = JSON.parse(
									result.function.arguments,
								);
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
			(request.tools
				?.map(tool => {
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
				.filter(tool => tool !== null) as any[]) || [];

		if ((request as any).handoffs && Array.isArray((request as any).handoffs)) {
			for (const handoff of (request as any).handoffs) {
				try {
					const handoffTool = this.convertHandoffTool(handoff);
					if (handoffTool) {
						ollamaTools.push(handoffTool);
					}
				} catch (e) {
					console.warn('🔍 Failed to convert handoff to tool:', e);
				}
			}
		}

		const chatOptions: any = {
			model: this.#model,
			messages: ollamaMessages,
			stream: stream as any,
		};

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
				reasoning_tokens:
					usage.completion_tokens_details?.reasoning_tokens || 0,
			},
		};
	}

	async getResponse(request: ModelRequest): Promise<ModelResponse> {
		const response = await withGenerationSpan(async span => {
			span.spanData.model = this.#model;
			span.spanData.model_config = request.modelSettings
				? {
						temperature: request.modelSettings.temperature,
						top_p: request.modelSettings.topP,
						frequency_penalty: request.modelSettings.frequencyPenalty,
						presence_penalty: request.modelSettings.presencePenalty,
				  }
				: {base_url: 'ollama_client'};
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
				const {content, ...rest} = message;
				output.push({
					id: response.id,
					type: 'message',
					role: 'assistant',
					content: [
						{
							type: 'output_text',
							text: content || '',
							providerData: rest,
						},
					],
					status: 'completed',
				});
			} else if (message.refusal) {
				const {refusal, ...rest} = message;
				output.push({
					id: response.id,
					type: 'message',
					role: 'assistant',
					content: [
						{
							type: 'refusal',
							refusal: refusal || '',
							providerData: rest,
						},
					],
					status: 'completed',
				});
			} else if (message.tool_calls) {
				for (const tool_call of message.tool_calls) {
					if (tool_call.type === 'function') {
						const {id: callId, ...remainingToolCallData} = tool_call;
						const {
							arguments: args,
							name,
							...remainingFunctionData
						} = tool_call.function;
						output.push({
							id: response.id,
							type: 'function_call',
							arguments: args,
							name: name,
							callId: callId,
							status: 'completed',
							providerData: {
								...remainingToolCallData,
								...remainingFunctionData,
							},
						});
					}
				}
			}
		}

		const modelResponse: ModelResponse = {
			usage: response.usage
				? new Usage(this.toResponseUsage(response.usage))
				: new Usage(),
			output,
			responseId: response.id,
			providerData: response,
		};

		return modelResponse;
	}

	async *getStreamedResponse(
		request: ModelRequest,
	): AsyncIterable<ResponseStreamEvent> {
		const span = request.tracing ? createGenerationSpan() : undefined;
		try {
			if (span) {
				span.start();
				setCurrentSpan(span);
			}
			const stream = await this.#fetchResponse(request, span, true);

			yield* this.convertOllamaStreamToResponses(
				stream,
				span,
				request.tracing === true,
			);
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
		span?: Span<GenerationSpanData>,
		tracingEnabled?: boolean,
	): AsyncIterable<ResponseStreamEvent> {
		let usage: any = undefined;
		let started = false;
		let accumulatedText = '';
		const responseId = generateCompletionId();

		for await (const chunk of stream) {
			if (!started) {
				started = true;
				yield {
					type: 'response_started',
					providerData: chunk,
				};
			}

			yield {
				type: 'model',
				event: chunk,
			};

			if (chunk.eval_count || chunk.prompt_eval_count) {
				usage = {
					prompt_tokens: chunk.prompt_eval_count || 0,
					completion_tokens: chunk.eval_count || 0,
					total_tokens:
						(chunk.prompt_eval_count || 0) + (chunk.eval_count || 0),
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
							tool_call.id &&
							typeof tool_call.id === 'string' &&
							tool_call.id.startsWith('call_')
								? tool_call.id
								: generateToolCallId();

						const functionCallEvent = {
							type: 'response_done',
							response: {
								id: responseId,
								usage: {
									inputTokens: usage?.prompt_tokens ?? 0,
									outputTokens: usage?.completion_tokens ?? 0,
									totalTokens: usage?.total_tokens ?? 0,
									inputTokensDetails: {cached_tokens: 0},
									outputTokensDetails: {reasoning_tokens: 0},
								},
								output: [
									{
										id: responseId,
										type: 'function_call',
										arguments: JSON.stringify(tool_call.function.arguments),
										name: tool_call.function.name,
										callId: callId,
										status: 'completed',
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
						id: responseId,
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

				yield {
					type: 'response_done',
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
