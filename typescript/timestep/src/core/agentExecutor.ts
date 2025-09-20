// @ts-nocheck
import {
	AgentExecutor,
	RequestContext,
	ExecutionEventBus,
	TaskStatusUpdateEvent,
	TaskArtifactUpdateEvent,
} from '@a2a-js/sdk/server';
// Note: new_agent_text_message may not be available, creating simple version
import {
	Agent,
	Runner,
	run,
	tool,
	setTracingExportApiKey,
	ModelProvider,
	Model,
	user,
	AgentHooks,
	AgentInputItem,
} from '@openai/agents';
import {
	setTracingDisabled,
	withTrace,
	getOrCreateTrace,
	TraceOptions,
	Trace,
	RunContext,
	withNewSpanContext,
	setCurrentSpan,
	resetCurrentSpan,
	RunHookEvents,
	getCurrentSpan,
} from '@openai/agents-core';
// fs and path imports removed - no longer needed for app config loading
import * as crypto from 'node:crypto';
import {AgentConfiguration, RunConfig} from '@openai/agents-core';
import {getGlobalTraceProvider} from '@openai/agents';
import {RunState, RunResult} from '@openai/agents-core';
// Note: These functions may not be available in the current package version
// Let's try importing from the main package instead
import {
	processModelResponse,
	executeToolsAndSideEffects,
} from '@openai/agents-core';
// Note: getTracingExportApiKey is not exported from the main package
// We'll create a simple workaround to track the API key
let _tracingApiKey: string | undefined = undefined;
// Note: TypeScript SDK has different API structure than Python
// Note: defineInputGuardrail not available, creating manual input guardrail
import {InputGuardrailTripwireTriggered} from '@openai/agents';
import {OllamaModel} from '../services/backing/models.js';
import {Ollama} from 'ollama';
import {TimestepAIModelProvider} from '../services/modelProvider.js';
import {AgentFactory} from '../services/agentFactory.js';
import {ContextService} from '../services/contextService.js';
import {
	RepositoryContainer,
	DefaultRepositoryContainer,
} from '../services/backing/repositoryContainer.js';

// App configuration is now loaded dynamically when needed via loadAppConfig() function

// Function to load model providers using the API
async function loadModelProviders(
	repositories?: RepositoryContainer,
): Promise<{[key: string]: any}> {
	try {
		const {listModelProviders} = await import('../api/modelProvidersApi.js');
		const response = await listModelProviders(repositories);
		const MODEL_PROVIDERS: {[key: string]: any} = {};

		for (const provider of response.data) {
			MODEL_PROVIDERS[provider.provider] = provider;
		}

		return MODEL_PROVIDERS;
	} catch (error) {
		console.warn(
			`Failed to load model providers: ${error}. Using empty configuration.`,
		);
		return {};
	}
}

// Tracing API key will be set when repositories are available
let tracing_api_key: string | undefined;
let tracing_api_key_set = false;

// Function to set tracing API key when repositories are available
async function setTracingApiKeyFromRepositories(
	repositories?: RepositoryContainer,
) {
	if (tracing_api_key_set) return;

	try {
		const providers = await loadModelProviders(repositories);
		tracing_api_key = providers.openai?.api_key;
		console.log(
			'🔑 Setting tracing export API key:',
			tracing_api_key ? `${tracing_api_key.substring(0, 10)}...` : 'undefined',
		);
		setTracingExportApiKey(tracing_api_key);

		// Store the API key locally for verification
		_tracingApiKey = tracing_api_key;
		console.log(
			'✅ Tracing export API key set successfully. Stored key:',
			_tracingApiKey ? `${_tracingApiKey.substring(0, 10)}...` : 'undefined',
		);
		tracing_api_key_set = true;
	} catch (error) {
		console.warn('Failed to load model providers for tracing:', error);
	}
}

// Function to check for tool call approval in message
function checkForToolCallApproval(
	message: any,
): {approved: boolean; decision: string; reason?: string} | null {
	if (!message.parts) return null;

	for (const part of message.parts) {
		// Handle structured tool approval data
		if (part.kind === 'data' && part.data?.toolCallResponse) {
			const response = part.data.toolCallResponse;
			return {
				approved: response.status === 'approved',
				decision: response.decision,
				reason: response.reason,
			};
		}
	}

	return null;
}

// MCP functions moved to agent_factory.ts

function createCompletedStatusUpdate(
	taskId: string,
	contextId: string,
	finalMessage?: string,
): TaskStatusUpdateEvent {
	const status: any = {
		state: 'completed',
		timestamp: new Date().toISOString(),
	};

	// Include finalMessage in the status if provided
	if (finalMessage) {
		status.result = finalMessage;
	}

	return {
		kind: 'status-update',
		taskId: taskId,
		contextId: contextId,
		status: status,
		final: true,
	};
}

function createInputRequiredStatusUpdate(
	taskId: string,
	contextId: string,
	interruptions?: any[],
): TaskStatusUpdateEvent {
	let messageText = 'Human approval required for tool execution';

	// Include tool call details in the message if interruptions are provided
	if (interruptions && interruptions.length > 0) {
		const toolCalls = interruptions
			.map(
				interruption =>
					`${interruption.rawItem.name}(${interruption.rawItem.arguments})`,
			)
			.join(', ');
		messageText = `Human approval required for tool execution: ${toolCalls}`;
	}

	return {
		kind: 'status-update',
		taskId: taskId,
		contextId: contextId,
		status: {
			state: 'input-required',
			message: {
				role: 'agent',
				parts: [
					{
						kind: 'text',
						text: messageText,
					},
				],
				messageId: crypto.randomUUID(),
				taskId: taskId,
				contextId: contextId,
			},
			timestamp: new Date().toISOString(),
		},
		final: false,
	};
}

function createToolCallArtifact(
	interruption: any,
	taskId: string,
	contextId: string,
): TaskArtifactUpdateEvent {
	const artifactId = `tool-call-${crypto.randomUUID()}`;

	return {
		kind: 'artifact-update',
		taskId: taskId,
		contextId: contextId,
		artifact: {
			artifactId: artifactId,
			name: 'Required Tool Call',
			description: `Tool call requiring approval: ${interruption.rawItem.name}`,
			parts: [
				{
					kind: 'data',
					data: {
						toolCall: {
							id: interruption.rawItem.callId || crypto.randomUUID(),
							name: interruption.rawItem.name,
							parameters: interruption.rawItem.arguments
								? JSON.parse(interruption.rawItem.arguments)
								: {},
						},
						authRequired: {
							type: 'approval',
							agent: interruption.agent.name,
							reason: 'Tool execution requires human approval',
						},
					},
				},
			],
		},
		final: false,
	};
}

// saveMessageToContext function removed - using SDK history format

// convertToSimpleFormat function removed - using SDK history format directly

async function getAgentInput(
	context: RequestContext,
	agent: Agent,
	contextService: ContextService,
): Promise<AgentInputItem[] | RunState> {
	const userMessage = context.userMessage;
	const contextId = context.contextId;
	const taskId = context.taskId;

	const isToolApproval = checkForToolCallApproval(userMessage);

	if (isToolApproval) {
		console.log(
			'🔍 User message is tool approval - loading saved state from context service',
		);
		const savedState = await contextService.getTaskState(contextId, taskId);
		if (!savedState) {
			throw new Error(
				`No saved state found for tool approval. ContextId: ${contextId}, TaskId: ${taskId}`,
			);
		}

		const runState = await RunState.fromString(
			agent,
			JSON.stringify(savedState),
		);

		console.log('✅ Tool call approved, approving interruption in saved state');
		const interruptions = runState.getInterruptions();
		if (!interruptions || interruptions.length === 0) {
			throw new Error(
				`No interruptions found in saved state for contextId: ${contextId}, taskId: ${taskId}`,
			);
		}

		runState.approve(interruptions[0]);
		return runState;
	} else {
		console.log(
			'🔍 User message is not tool approval - loading conversation history from context service',
		);

		const toolResponsePart = userMessage.parts?.find(
			part => part.kind === 'data' && part.data?.toolCallResponse,
		);

		if (toolResponsePart) {
			throw new Error(
				'Tool response detected but not handled as tool approval - this should not happen',
			);
		}

		const messageText =
			userMessage.parts?.[0]?.text || userMessage.text || 'Hello';

		let history = await contextService.getTaskHistory(contextId, taskId);

		history.push(user(messageText));

		return history;
	}
}

// Configuration interfaces
export interface ContextRepositoryOptions {
	// Generic options that could apply to any repository type
}

export interface AgentExecutorConfig {
	repositories?: RepositoryContainer;
}

// getContext function and AgentFactory class moved to agent_factory.ts

export class TimestepAIAgentExecutor implements AgentExecutor {
	agentFactory: AgentFactory;
	contextService: ContextService;
	repositories: RepositoryContainer;

	constructor({repositories}: AgentExecutorConfig = {}) {
		// Check if we're in a restricted environment (Supabase Edge Functions)
		const isRestrictedEnvironment =
			typeof Deno !== 'undefined' && Deno.env.get('DENO_DEPLOYMENT_ID');

		if (repositories) {
			this.repositories = repositories;
		} else if (isRestrictedEnvironment) {
			// In restricted environments, throw an error if no repositories provided
			throw new Error(
				'Custom repositories must be provided in restricted environments (Supabase Edge Functions). Cannot use default file-based repositories.',
			);
		} else {
			this.repositories = new DefaultRepositoryContainer();
		}

		this.agentFactory = new AgentFactory(this.repositories);
		this.contextService = new ContextService(this.repositories.contexts);

		// Set tracing API key from repositories (async, but don't wait)
		setTracingApiKeyFromRepositories(this.repositories).catch(error => {
			console.warn('Failed to set tracing API key:', error);
		});
	}

	async execute(
		context: RequestContext,
		eventBus: ExecutionEventBus,
	): Promise<void> {
		// const trace: string | Trace = "Timestep AI Agent Execution"

		const traceId = `trace_${context.taskId.replace(/-/g, '')}`; // Use taskId as traceId for unified tracing, prefixed with trace_ and dashes removed

		const traceOptions: TraceOptions = {
			traceId: traceId,
		};

		const trace: Trace = getGlobalTraceProvider().createTrace(traceOptions);

		try {
			await withTrace(trace, async trace => {
				// 1. Publish initial Task if it doesn't exist
				const userMessage = context.userMessage;
				const existingTask = context.task;
				const taskId = context.taskId;
				const contextId = context.contextId;

				if (!existingTask) {
					const initialTask: any = {
						kind: 'task',
						id: taskId,
						contextId: contextId,
						status: {
							state: 'submitted',
							timestamp: new Date().toISOString(),
						},
						history: [userMessage],
						metadata: userMessage.metadata,
					};
					eventBus.publish(initialTask);
				}

				// 2. Publish working status
				const workingStatusUpdate: any = {
					kind: 'status-update',
					taskId: taskId,
					contextId: contextId,
					status: {
						state: 'working',
						timestamp: new Date().toISOString(),
					},
					final: false,
				};
				eventBus.publish(workingStatusUpdate);

				const runConfig: RunConfig = {
					modelProvider: new TimestepAIModelProvider(this.repositories),
					groupId: contextId,
					traceId: trace.traceId, // Since we're associating the traceId with the run, then the history will be associated with the trace (task)
					traceIncludeSensitiveData: true,
					tracingDisabled: false,
				};

				const runner = new Runner(runConfig);

				// Load the existing context to get the agentId
				const contextObj = await this.contextService.repository.load(contextId);
				if (!contextObj) {
					throw new Error(
						`Context ${contextId} not found - it should have been created by the A2A server`,
					);
				}
				const agentId = contextObj.agentId;

				const agentConfigResult = await this.agentFactory.buildAgentConfig(
					agentId,
				);
				const agent = agentConfigResult.createAgent();

				// Ensure the context has this task
				if (!contextObj.getTask(taskId)) {
					await this.contextService.addTask(contextId, {
						id: taskId,
						contextId: contextId,
						kind: 'task',
						status: {state: 'submitted', timestamp: new Date().toISOString()},
						history: [context.userMessage],
						metadata: context.userMessage.metadata,
					});
				}

				const agentInput = await getAgentInput(
					context,
					agent,
					this.contextService,
				);

				const result = await runner.run(agent, agentInput);

				const hasInterruptions = result.interruptions?.length > 0;

				if (hasInterruptions) {
					console.log('Interruptions found, saving state to context service');

					await this.contextService.updateFromRunResult(
						contextId,
						taskId,
						result,
					);

					const inputRequiredStatusUpdate = createInputRequiredStatusUpdate(
						taskId,
						context.contextId,
						result.interruptions,
					);
					console.log(
						'🔍 Publishing input-required status update:',
						JSON.stringify(inputRequiredStatusUpdate, null, 2),
					);
					eventBus.publish(inputRequiredStatusUpdate);

					eventBus.finished();
					return;
				} else {
					console.log('No interruptions, saving history to context service');

					console.log(`[${agent.name}] ${result.finalOutput}`);

					await this.contextService.updateFromRunResult(
						contextId,
						taskId,
						result,
					);
				}

				const finalOutput = result.finalOutput || '';

				// For task-generating agents, publish the completed Task object (not just status)
				// Retrieve the updated task from the context service
				const completedTask = await this.contextService.getTask(
					context.contextId,
					taskId,
				);
				if (completedTask) {
					// Update the task status to completed and add the final output
					completedTask.status = {
						state: 'completed',
						timestamp: new Date().toISOString(),
					};

					// Add the final result to task.status.message (expected by client)
					// Client expects a Message object with parts array, not a plain string
					if (finalOutput) {
						completedTask.status.message = {
							messageId: crypto.randomUUID(),
							kind: 'message',
							role: 'agent',
							parts: [
								{
									kind: 'text',
									text: finalOutput,
								},
							],
							contextId: context.contextId,
							timestamp: new Date().toISOString(),
						};
					}

					console.log(
						'🔍 Publishing completed task:',
						JSON.stringify(completedTask, null, 2),
					);
					eventBus.publish(completedTask);
				} else {
					console.warn(
						'🔍 Could not retrieve completed task, falling back to status update',
					);
					const completedStatusUpdate = createCompletedStatusUpdate(
						taskId,
						context.contextId,
						finalOutput,
					);
					console.log(
						'🔍 Publishing completed status update:',
						JSON.stringify(completedStatusUpdate, null, 2),
					);
					eventBus.publish(completedStatusUpdate);
				}

				// Signal that the stream is complete
				eventBus.finished();
			});
		} catch (e) {
			console.error('🔍 Error in execution:', e);
			throw e;
		} finally {
			// Flush any remaining traces
			getGlobalTraceProvider().forceFlush();
		}
	}

	async cancelTask(taskId: string, eventBus: ExecutionEventBus): Promise<void> {
		throw new Error('cancel not supported');
	}
}
