import {Task} from '@a2a-js/sdk';
import {AgentInputItem} from '@openai/agents';
import {RunResult} from '@openai/agents-core';

// Simplified context class that bridges A2A protocol with OpenAI Agents
export class Context {
	contextId: string;
	agentId: string;
	// Task-based storage to align with A2A protocol
	taskHistories: Record<string, AgentInputItem[]>; // taskId -> current conversation history
	taskStates: Record<string, any>; // taskId -> serialized agent state
	tasks: Task[]; // A2A tasks for reference

	constructor(contextId: string, agentId: string) {
		this.contextId = contextId;
		this.agentId = agentId;
		this.taskHistories = {};
		this.taskStates = {};
		this.tasks = [];
	}

	// Get conversation history for a specific task (for OpenAI Agents run input)
	getTaskHistory(taskId: string): AgentInputItem[] {
		return this.taskHistories[taskId] || [];
	}

	// Get full conversation history across all tasks in chronological order
	getFullConversationHistory(): AgentInputItem[] {
		const fullHistory: AgentInputItem[] = [];

		// Sort tasks by creation time (using task ID timestamp if available, or order in array)
		const sortedTasks = [...this.tasks].sort((a, b) => {
			// If tasks have timestamps in metadata, use those
			const aTime = String(
				a.metadata?.['timestamp'] || a.status?.timestamp || '0',
			);
			const bTime = String(
				b.metadata?.['timestamp'] || b.status?.timestamp || '0',
			);
			return aTime.localeCompare(bTime);
		});

		// Collect history from all tasks in chronological order
		for (const task of sortedTasks) {
			const taskHistory = this.taskHistories[task.id];
			if (taskHistory && taskHistory.length > 0) {
				fullHistory.push(...taskHistory);
			}
		}

		return fullHistory;
	}

	// Set conversation history for a task (from OpenAI Agents run result)
	setTaskHistory(taskId: string, history: AgentInputItem[]): void {
		this.taskHistories[taskId] = structuredClone(history);
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
		this.setTaskHistory(taskId, result.history);
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
