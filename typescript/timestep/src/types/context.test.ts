import {describe, it, expect, beforeEach} from 'vitest';
import {Context} from './context.js';
import {Task, Message} from '@a2a-js/sdk';

describe('Context', () => {
	let context: Context;
	const contextId = 'test-context-id';
	const agentId = 'test-agent-id';

	beforeEach(() => {
		context = new Context(contextId, agentId);
	});

	describe('constructor', () => {
		it('should initialize with provided values', () => {
			expect(context.contextId).toBe(contextId);
			expect(context.agentId).toBe(agentId);
			expect(context.taskHistories).toEqual({});
			expect(context.taskStates).toEqual({});
			expect(context.tasks).toEqual([]);
		});
	});

	describe('getTaskHistory', () => {
		it('should return empty array for non-existent task', () => {
			const result = context.getTaskHistory('non-existent-task');
			expect(result).toEqual([]);
		});

		it('should return task history when it exists', () => {
			const mockHistory: Message[] = [
				{
					messageId: 'msg-1',
					kind: 'message',
					role: 'user',
					parts: [{kind: 'text', text: 'Hello'}],
					contextId: contextId,
				},
			];
			context.taskHistories['task-1'] = mockHistory;

			const result = context.getTaskHistory('task-1');
			expect(result).toEqual(mockHistory);
		});
	});

	describe('setTaskHistory', () => {
		it('should set task history with structured clone', () => {
			const mockHistory: Message[] = [
				{
					messageId: 'msg-1',
					kind: 'message',
					role: 'user',
					parts: [{kind: 'text', text: 'Hello'}],
					contextId: contextId,
				},
			];

			context.setTaskHistory('task-1', mockHistory);

			expect(context.taskHistories['task-1']).toEqual(mockHistory);
			expect(context.taskHistories['task-1']).not.toBe(mockHistory); // Should be cloned
		});
	});

	describe('addMessageToTaskHistory', () => {
		it('should create new history array and add message', () => {
			const mockMessage: Message = {
				messageId: 'msg-1',
				kind: 'message',
				role: 'user',
				parts: [{kind: 'text', text: 'Hello'}],
				contextId: contextId,
			};

			context.addMessageToTaskHistory('task-1', mockMessage);

			expect(context.taskHistories['task-1']).toEqual([mockMessage]);
		});

		it('should append to existing history', () => {
			const existingMessage: Message = {
				messageId: 'msg-1',
				kind: 'message',
				role: 'user',
				parts: [{kind: 'text', text: 'Hello'}],
				contextId: contextId,
			};
			const newMessage: Message = {
				messageId: 'msg-2',
				kind: 'message',
				role: 'agent',
				parts: [{kind: 'text', text: 'Hi there!'}],
				contextId: contextId,
			};

			context.taskHistories['task-1'] = [existingMessage];
			context.addMessageToTaskHistory('task-1', newMessage);

			expect(context.taskHistories['task-1']).toEqual([
				existingMessage,
				newMessage,
			]);
		});
	});

	describe('getTaskState', () => {
		it('should return undefined for non-existent task', () => {
			const result = context.getTaskState('non-existent-task');
			expect(result).toBeUndefined();
		});

		it('should return task state when it exists', () => {
			const mockState = {status: 'active', data: 'test'};
			context.taskStates['task-1'] = mockState;

			const result = context.getTaskState('task-1');
			expect(result).toEqual(mockState);
		});
	});

	describe('setTaskState', () => {
		it('should set task state with JSON serialization', () => {
			const mockState = {status: 'active', data: 'test'};

			context.setTaskState('task-1', mockState);

			expect(context.taskStates['task-1']).toEqual(mockState);
		});

		it('should handle complex objects with JSON serialization', () => {
			const complexState = {
				status: 'active',
				nested: {value: 123, array: [1, 2, 3]},
				date: new Date('2023-01-01'),
			};

			context.setTaskState('task-1', complexState);

			// Should be serialized/deserialized
			expect(context.taskStates['task-1']).toEqual(
				JSON.parse(JSON.stringify(complexState)),
			);
		});
	});

	describe('updateFromRunResult', () => {
		it('should update task history and state from run result', () => {
			const mockHistory = [
				{role: 'user', content: 'Hello'},
				{role: 'assistant', content: 'Hi!'},
			];
			const mockState = {status: 'completed'};
			const mockRunResult = {
				history: mockHistory,
				state: mockState,
			} as any;

			context.updateFromRunResult('task-1', mockRunResult);

			expect(context.taskHistories['task-1']).toEqual(mockHistory);
			expect(context.taskStates['task-1']).toEqual(mockState);
		});

		it('should handle run result without state', () => {
			const mockHistory = [{role: 'user', content: 'Hello'}];
			const mockRunResult = {
				history: mockHistory,
			} as any;

			context.updateFromRunResult('task-1', mockRunResult);

			expect(context.taskHistories['task-1']).toEqual(mockHistory);
			expect(context.taskStates['task-1']).toBeUndefined();
		});
	});

	describe('addTask', () => {
		it('should add new task to tasks array', () => {
			const mockTask: Task = {
				kind: 'task',
				id: 'task-1',
				contextId: contextId,
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};

			context.addTask(mockTask);

			expect(context.tasks).toEqual([mockTask]);
		});

		it('should throw error when adding duplicate task ID', () => {
			const mockTask: Task = {
				kind: 'task',
				id: 'task-1',
				contextId: contextId,
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};

			context.addTask(mockTask);

			expect(() => context.addTask(mockTask)).toThrow(
				'Task with ID task-1 already exists in context test-context-id',
			);
		});
	});

	describe('updateTask', () => {
		it('should update existing task', () => {
			const originalTask: Task = {
				kind: 'task',
				id: 'task-1',
				contextId: contextId,
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};
			const updatedTask: Task = {
				kind: 'task',
				id: 'task-1',
				contextId: contextId,
				status: {state: 'completed', timestamp: new Date().toISOString()},
				history: [],
			};

			context.addTask(originalTask);
			context.updateTask(updatedTask);

			expect(context.tasks).toEqual([updatedTask]);
		});

		it('should add task if it does not exist', () => {
			const newTask: Task = {
				kind: 'task',
				id: 'task-1',
				contextId: contextId,
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};

			context.updateTask(newTask);

			expect(context.tasks).toEqual([newTask]);
		});
	});

	describe('getTask', () => {
		it('should return undefined for non-existent task', () => {
			const result = context.getTask('non-existent-task');
			expect(result).toBeUndefined();
		});

		it('should return task when it exists', () => {
			const mockTask: Task = {
				kind: 'task',
				id: 'task-1',
				contextId: contextId,
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};

			context.addTask(mockTask);
			const result = context.getTask('task-1');

			expect(result).toEqual(mockTask);
		});
	});

	describe('toJSON', () => {
		it('should serialize context to JSON', () => {
			const mockTask: Task = {
				kind: 'task',
				id: 'task-1',
				contextId: contextId,
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};
			const mockMessage: Message = {
				messageId: 'msg-1',
				kind: 'message',
				role: 'user',
				parts: [{kind: 'text', text: 'Hello'}],
				contextId: contextId,
			};

			context.addTask(mockTask);
			context.addMessageToTaskHistory('task-1', mockMessage);
			context.setTaskState('task-1', {status: 'active'});

			const result = context.toJSON();

			expect(result).toEqual({
				contextId: contextId,
				agentId: agentId,
				taskHistories: {'task-1': [mockMessage]},
				taskStates: {'task-1': {status: 'active'}},
				tasks: [mockTask],
			});
		});
	});

	describe('fromJSON', () => {
		it('should create context from JSON data', () => {
			const mockTask: Task = {
				kind: 'task',
				id: 'task-1',
				contextId: contextId,
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};
			const mockMessage: Message = {
				messageId: 'msg-1',
				kind: 'message',
				role: 'user',
				parts: [{kind: 'text', text: 'Hello'}],
				contextId: contextId,
			};

			const jsonData = {
				contextId: contextId,
				agentId: agentId,
				taskHistories: {'task-1': [mockMessage]},
				taskStates: {'task-1': {status: 'active'}},
				tasks: [mockTask],
			};

			const result = Context.fromJSON(jsonData);

			expect(result.contextId).toBe(contextId);
			expect(result.agentId).toBe(agentId);
			expect(result.taskHistories).toEqual({'task-1': [mockMessage]});
			expect(result.taskStates).toEqual({'task-1': {status: 'active'}});
			expect(result.tasks).toEqual([mockTask]);
		});

		it('should handle missing optional fields', () => {
			const jsonData = {
				contextId: contextId,
				agentId: agentId,
			};

			const result = Context.fromJSON(jsonData);

			expect(result.contextId).toBe(contextId);
			expect(result.agentId).toBe(agentId);
			expect(result.taskHistories).toEqual({});
			expect(result.taskStates).toEqual({});
			expect(result.tasks).toEqual([]);
		});
	});
});
