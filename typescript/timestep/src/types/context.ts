import {Task, Message} from '@a2a-js/sdk';
import {RunResult} from '@openai/agents-core';

// Simplified context class that bridges A2A protocol with OpenAI Agents
export class Context {
	contextId: string;
	agentId: string;
	// Task-based storage to align with A2A protocol
	taskHistories: Record<string, Message[]>; // taskId -> A2A conversation history
	taskStates: Record<string, any>; // taskId -> serialized agent state
	tasks: Task[]; // A2A tasks for reference

	constructor(contextId: string, agentId: string) {
		this.contextId = contextId;
		this.agentId = agentId;
		this.taskHistories = {};
		this.taskStates = {};
		this.tasks = [];
	}

	// Get conversation history for a specific task (returns A2A Messages)
	getTaskHistory(taskId: string): Message[] {
		return this.taskHistories[taskId] || [];
	}

	// Set conversation history for a task (from A2A Messages)
	setTaskHistory(taskId: string, history: Message[]): void {
		this.taskHistories[taskId] = structuredClone(history);
	}

	// Add a message to task history
	addMessageToTaskHistory(taskId: string, message: Message): void {
		if (!this.taskHistories[taskId]) {
			this.taskHistories[taskId] = [];
		}
		this.taskHistories[taskId].push(message);
	}

	// Get agent state for a specific task (for OpenAI Agents run input)
	getTaskState(taskId: string): any {
		return this.taskStates[taskId];
	}

	// Set agent state for a task (from OpenAI Agents run result)
	setTaskState(taskId: string, state: any): void {
		// Use JSON serialization for agent state since it may contain functions
		this.taskStates[taskId] = JSON.parse(JSON.stringify(state));
	}

	// Update from OpenAI Agents run result
	updateFromRunResult(taskId: string, result: RunResult<unknown, any>): void {
		// Convert Agents SDK history to A2A Messages
		// Note: This will need to be implemented with proper mapping
		// For now, we'll store the raw history and convert when needed
		this.setTaskHistory(taskId, result.history as any); // TODO: Implement proper conversion
		if (result.state) {
			this.setTaskState(taskId, result.state);
		}
	}

	// Add A2A task reference
	addTask(task: Task): void {
		// Check if task ID already exists
		if (this.tasks.some(t => t.id === task.id)) {
			throw new Error(
				`Task with ID ${task.id} already exists in context ${this.contextId}`,
			);
		}
		this.tasks.push(task);
	}

	// Update an existing A2A task
	updateTask(task: Task): void {
		const existingIndex = this.tasks.findIndex(t => t.id === task.id);
		if (existingIndex >= 0) {
			this.tasks[existingIndex] = task;
		} else {
			// If task doesn't exist, add it
			this.tasks.push(task);
		}
	}

	// Get A2A task by ID
	getTask(taskId: string): Task | undefined {
		return this.tasks.find(t => t.id === taskId);
	}

	// Serialize for storage
	toJSON(): any {
		return {
			contextId: this.contextId,
			agentId: this.agentId,
			taskHistories: this.taskHistories,
			taskStates: this.taskStates,
			tasks: this.tasks,
		};
	}

	// Create from JSON
	static fromJSON(data: any): Context {
		const context = new Context(data.contextId, data.agentId);
		context.taskHistories = data.taskHistories || {};
		context.taskStates = data.taskStates || {};
		context.tasks = data.tasks || [];
		return context;
	}
}
