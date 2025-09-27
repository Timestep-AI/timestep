import {Task, Message} from '@a2a-js/sdk';
import {RunResult} from '@openai/agents-core';
import {Context} from '../types/context.js';
import {Repository} from './backing/repository.js';

/**
 * Service for managing context operations.
 * Handles business logic for context management and delegates persistence to repository.
 */
export class ContextService {
	constructor(private repository: Repository<Context, string>) {}

	/**
	 * List all contexts
	 */
	async listContexts(): Promise<Context[]> {
		return await this.repository.list();
	}

	/**
	 * Get a specific context by ID
	 */
	async getContext(contextId: string): Promise<Context | null> {
		return await this.repository.load(contextId);
	}

	/**
	 * Get or create a context by ID
	 */
	async getOrCreate(_contextId: string): Promise<Context> {
		throw new Error(
			'getOrCreate without agentId not yet implemented - contexts should be created by A2A server',
		);
	}

	/**
	 * Update context with results from an OpenAI Agents run
	 */
	async updateFromRunResult(
		contextId: string,
		taskId: string,
		result: RunResult<unknown, any>,
	): Promise<void> {
		const context = await this.repository.load(contextId);
		if (!context) {
			throw new Error(`Context ${contextId} not found`);
		}

		context.updateFromRunResult(taskId, result);
		await this.repository.save(context);
	}

	/**
	 * Get conversation history for a specific task (returns A2A Messages)
	 */
	async getTaskHistory(contextId: string, taskId: string): Promise<Message[]> {
		const context = await this.repository.load(contextId);
		return context?.getTaskHistory(taskId) || [];
	}

	/**
	 * Get agent state for a specific task
	 */
	async getTaskState(contextId: string, taskId: string): Promise<any> {
		const context = await this.repository.load(contextId);
		return context?.getTaskState(taskId);
	}

	/**
	 * Add a task to a context
	 */
	async addTask(contextId: string, task: Task): Promise<void> {
		const context = await this.repository.load(contextId);
		if (!context) {
			throw new Error(`Context ${contextId} not found`);
		}

		context.addTask(task);
		await this.repository.save(context);
	}

	/**
	 * Update an existing task in a context
	 */
	async updateTask(contextId: string, task: Task): Promise<void> {
		const context = await this.repository.load(contextId);
		if (!context) {
			throw new Error(`Context ${contextId} not found`);
		}

		context.updateTask(task);
		await this.repository.save(context);
	}

	/**
	 * Get a task from a context
	 */
	async getTask(contextId: string, taskId: string): Promise<Task | undefined> {
		const context = await this.repository.load(contextId);
		return context?.getTask(taskId);
	}

	/**
	 * Add a message to task history
	 */
	async addMessageToTaskHistory(
		contextId: string,
		taskId: string,
		message: Message,
	): Promise<void> {
		const context = await this.repository.load(contextId);
		if (!context) {
			throw new Error(`Context ${contextId} not found`);
		}

		context.addMessageToTaskHistory(taskId, message);
		await this.repository.save(context);
	}

	/**
	 * Save a context
	 */
	async save(context: Context): Promise<void> {
		await this.repository.save(context);
	}
}
