// @ts-nocheck
import {
	AgentExecutor,
	RequestContext,
	ExecutionEventBus,
	Task,
	TaskStatusUpdateEvent,
	TaskArtifactUpdateEvent,
	Message,
	TextPart,
} from '@a2a-js/sdk/server';
import {
	TaskState,
} from '@a2a-js/sdk';

// Define the TaskState values we're using
const TASK_STATES = {
	SUBMITTED: 'submitted' as TaskState,
	WORKING: 'working' as TaskState,
	INPUT_REQUIRED: 'input-required' as TaskState,
	COMPLETED: 'completed' as TaskState,
	FAILED: 'failed' as TaskState,
	CANCELED: 'canceled' as TaskState,
	REJECTED: 'rejected' as TaskState,
} as const;
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
	assistant,
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
	RunHandoffCallItem,
	RunHandoffOutputItem,
	RunToolCallItem,
	RunToolCallOutputItem,
	RunReasoningItem,
	RunToolApprovalItem,
	RunMessageOutputItem,
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

// Mapping functions between A2A protocol and Agents SDK formats

/**
 * Converts A2A Message to Agents SDK AgentInputItem
 */
function a2aMessageToAgentInputItem(message: Message): AgentInputItem {
	if (message.role === 'user') {
		// Extract text from message parts
		const textParts = message.parts.filter((p): p is TextPart => p.kind === 'text');
		const text = textParts.map(p => p.text).join(' ');
		return user(text);
	} else if (message.role === 'agent') {
		// For agent messages, we need to create an assistant message
		const textParts = message.parts.filter((p): p is TextPart => p.kind === 'text');
		const text = textParts.map(p => p.text).join(' ');
		return assistant(text);
	}
	throw new Error(`Unsupported message role: ${message.role}`);
}

/**
 * Converts Agents SDK AgentInputItem to A2A Message
 */
function agentInputItemToA2aMessage(inputItem: AgentInputItem, taskId: string, contextId: string): Message {
	const messageId = crypto.randomUUID();
	
	if (inputItem.role === 'user') {
		return {
			kind: 'message',
			role: 'user',
			messageId: messageId,
			parts: [{ kind: 'text', text: inputItem.content }],
			taskId: taskId,
			contextId: contextId,
			timestamp: new Date().toISOString(),
		};
	} else if (inputItem.role === 'assistant') {
		return {
			kind: 'message',
			role: 'agent',
			messageId: messageId,
			parts: [{ kind: 'text', text: inputItem.content }],
			taskId: taskId,
			contextId: contextId,
			timestamp: new Date().toISOString(),
		};
	}
	throw new Error(`Unsupported input item role: ${inputItem.role}`);
}

/**
 * Converts array of A2A Messages to Agents SDK AgentInputItem array
 */
function a2aMessagesToAgentInputItems(messages: Message[]): AgentInputItem[] {
	return messages.map(msg => a2aMessageToAgentInputItem(msg));
}

/**
 * Converts array of Agents SDK AgentInputItem to A2A Messages
 */
function agentInputItemsToA2aMessages(inputItems: AgentInputItem[], taskId: string, contextId: string): Message[] {
	return inputItems.map(item => agentInputItemToA2aMessage(item, taskId, contextId));
}

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
		state: TASK_STATES.COMPLETED,
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

async function getAgentInput(
	context: RequestContext,
	agent: Agent,
	contextService: ContextService,
): Promise<AgentInputItem[] | RunState> {
	const userMessage = context.userMessage;
	const contextId = context.contextId;
	const taskId = context.taskId;

	// Check if the task is already completed (Task-generating Agents approach)
	const existingTask = await contextService.getTask(contextId, taskId);
	if (existingTask && existingTask.status.state === TASK_STATES.COMPLETED) {
		throw new Error(
			`Task ${taskId} is already completed. No more messages can be sent to completed tasks.`,
		);
	}

	const isToolApproval = checkForToolCallApproval(userMessage);

	if (isToolApproval) {
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

		const interruptions = runState.getInterruptions();

		if (!interruptions || interruptions.length === 0) {
			throw new Error(
				`No interruptions found in saved state for contextId: ${contextId}, taskId: ${taskId}`,
			);
		}

		runState.approve(interruptions[0]);

		return runState;
	} else {
		const toolResponsePart = userMessage.parts?.find(
			part => part.kind === 'data' && part.data?.toolCallResponse,
		);

		if (toolResponsePart) {
			throw new Error(
				'Tool response detected but not handled as tool approval - this should not happen',
			);
		}

		// Get A2A message history and convert to Agents SDK format
		const a2aHistory = await contextService.getTaskHistory(contextId, taskId);
		const agentInputHistory = a2aMessagesToAgentInputItems(a2aHistory);

		// Add the current user message to the history
		const currentUserInput = a2aMessageToAgentInputItem(userMessage);
		agentInputHistory.push(currentUserInput);

		return agentInputHistory;
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

	/**
	 * Creates and publishes an initial task
	 */
	private async createAndPublishInitialTask(
		context: RequestContext,
		eventBus: ExecutionEventBus,
	): Promise<void> {
		const userMessage = context.userMessage;
		const taskId = context.taskId;
		const contextId = context.contextId;

		// Get the context to check for existing tasks
		const contextObj = await this.contextService.getContext(contextId);
		let history: Message[] = [userMessage];

		if (contextObj && contextObj.tasks.length > 0) {
			// Get the most recent task
			const mostRecentTask = contextObj.tasks[contextObj.tasks.length - 1];
			if (mostRecentTask) {
				// Get the history from the most recent task
				const recentTaskHistory = contextObj.getTaskHistory(mostRecentTask.id);
				if (recentTaskHistory && recentTaskHistory.length > 0) {
					// Combine the recent task history with the new user message
					history = [...recentTaskHistory, userMessage];
				}
			}
		}

		const now = new Date().toISOString();
		const initialTask: Task = {
			kind: 'task',
			id: taskId,
			contextId: contextId,
			status: {
				state: TASK_STATES.SUBMITTED,
				timestamp: now,
			},
			history: history,
			metadata: userMessage.metadata,
			createdAt: now,
			updatedAt: now,
		};

		// Persist the task before publishing
		await this.contextService.addTask(contextId, initialTask);
		
		// Also persist the user message to the task history
		await this.contextService.addMessageToTaskHistory(contextId, taskId, userMessage);
		
		eventBus.publish(initialTask);
	}

	/**
	 * Creates and publishes a status update
	 */
	private createAndPublishStatusUpdate(
		taskId: string,
		contextId: string,
		state: TaskState,
		eventBus: ExecutionEventBus,
		message?: any,
		final: boolean = false,
	): void {
		const statusUpdate: TaskStatusUpdateEvent = {
			kind: 'status-update',
			taskId: taskId,
			contextId: contextId,
			status: {
				state: state,
				timestamp: new Date().toISOString(),
				...(message && { message }),
			},
			final: final,
		};
		eventBus.publish(statusUpdate);
	}

	/**
	 * Creates and publishes a working status update
	 */
	private createAndPublishWorkingStatus(
		taskId: string,
		contextId: string,
		eventBus: ExecutionEventBus,
	): void {
		this.createAndPublishStatusUpdate(taskId, contextId, TASK_STATES.WORKING, eventBus);
	}

	/**
	 * Creates and publishes a status update with a message
	 */
	private async createAndPublishStatusWithMessage(
		taskId: string,
		contextId: string,
		state: TaskState,
		messageData: any,
		eventBus: ExecutionEventBus,
		final: boolean = false,
	): Promise<void> {
		const message: Message = {
			messageId: crypto.randomUUID(),
			kind: 'message',
			role: 'agent',
			parts: [
				{
					kind: 'data',
					data: messageData,
				},
			],
			contextId: contextId,
			taskId: taskId,
			timestamp: new Date().toISOString(),
		};

		// Persist the message to task history
		await this.contextService.addMessageToTaskHistory(contextId, taskId, message);

		this.createAndPublishStatusUpdate(taskId, contextId, state, eventBus, message, final);
	}

	/**
	 * Creates and publishes a completed task (Task-generating Agents approach)
	 * Always publishes a Task object, never just a status update
	 */
	private async createAndPublishCompletedTask(
		context: RequestContext,
		taskId: string,
		finalOutput: string,
		eventBus: ExecutionEventBus,
	): Promise<void> {
		// Get the current task from the context service
		const currentTask = await this.contextService.getTask(
			context.contextId,
			taskId,
		);

		if (!currentTask) {
			throw new Error(`Task ${taskId} not found in context ${context.contextId}`);
		}

		// Create the final agent message if there's output
		let finalMessage: Message | undefined;
		if (finalOutput) {
			finalMessage = {
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
				taskId: taskId,
				timestamp: new Date().toISOString(),
			};

			// Persist the final message to task history
			await this.contextService.addMessageToTaskHistory(
				context.contextId,
				taskId,
				finalMessage,
			);
		}

		// Update the task to completed state
		const now = new Date().toISOString();
		const completedTask: Task = {
			...currentTask,
			status: {
				state: TASK_STATES.COMPLETED,
				timestamp: now,
				...(finalMessage && { message: finalMessage }),
			},
			updatedAt: now,
		};

		// Update the task in the context service
		await this.contextService.updateTask(context.contextId, completedTask);

		// Publish the completed task (Task-generating Agents approach)
		eventBus.publish(completedTask);
	}

	async execute(
		context: RequestContext,
		eventBus: ExecutionEventBus,
	): Promise<void> {
		const traceId = `trace_${context.taskId.replace(/-/g, '')}`; // Use taskId as traceId for unified tracing, prefixed with trace_ and dashes removed

		const traceOptions: TraceOptions = {
			traceId: traceId,
		};

		const trace: Trace = getGlobalTraceProvider().createTrace(traceOptions);

		try {
			await withTrace(trace, async trace => {
				const existingTask = context.task;
				const taskId = context.taskId;
				const contextId = context.contextId;

				if (!existingTask) {
					await this.createAndPublishInitialTask(context, eventBus);
				}

				this.createAndPublishWorkingStatus(taskId, contextId, eventBus);

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

				const agentInput = await getAgentInput(
					context,
					agent,
					this.contextService,
				);

				// Use streaming mode from the Agents SDK
				const stream = await runner.run(agent, agentInput, {stream: true});

				for await (const event of stream) {
					// these are the raw events from the model
					if (event.type === 'raw_model_stream_event') {
						// Only publish streaming events for text output, suppress other raw events
						if (event.data?.type === 'output_text_delta' && event.data?.delta) {
							await this.createAndPublishStatusWithMessage(
								taskId,
								contextId,
								TASK_STATES.WORKING,
								event.data,
								eventBus,
							);
						}
					}
					// agent updated events
					// if (event.type === 'agent_updated_stream_event') {
					// console.log(`${event.type} %s`, event.agent.name);
					// }
					// Agent SDK specific events
					// if (event.type === 'run_item_stream_event') {
					// console.log(`${event.type} %o`, event.item);
					// }
					if (event.type === 'run_item_stream_event') {
						// Only publish important events, suppress verbose console logging
						if (
							event.name === 'handoff_occurred' ||
							event.name === 'handoff_requested' ||
							event.name === 'tool_called' ||
							event.name === 'tool_approval_requested' ||
							event.name === 'tool_output'
						) {
							const state = event.name === 'tool_approval_requested' ? TASK_STATES.INPUT_REQUIRED : TASK_STATES.WORKING;

							if (event.name === 'tool_approval_requested') {
								await this.contextService.updateFromRunResult(
									contextId,
									taskId,
									stream,
								);

								await this.createAndPublishStatusWithMessage(
									taskId,
									contextId,
									state,
									event.item.rawItem,
									eventBus,
								);
								eventBus.finished();
								return;
							}

							await this.createAndPublishStatusWithMessage(
								taskId,
								contextId,
								state,
								event.item.rawItem,
								eventBus,
							);
						}
					}
				}

				// Wait for the stream to complete
				await stream.completed;

				// Pass the StreamedRunResult directly since it implements the RunResult interface
				await this.contextService.updateFromRunResult(
					contextId,
					taskId,
					stream,
				);

				// Extract final output from the stream result
				// The stream result should contain the final agent response
				const finalOutput = stream.text || ''; // Get the final text output from the stream

				// For task-generating agents, publish the completed Task object (not just status)
				await this.createAndPublishCompletedTask(context, taskId, finalOutput, eventBus);

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
