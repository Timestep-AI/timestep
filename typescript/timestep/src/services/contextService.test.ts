import {describe, it, expect, vi, beforeEach} from 'vitest';
import {ContextService} from './contextService.js';
import {Task, Message} from '@a2a-js/sdk';

// Mock the Context class
vi.mock('../types/context.js', () => ({
	Context: vi.fn().mockImplementation(() => ({
		getTaskHistory: vi.fn().mockReturnValue([]),
		getTaskState: vi.fn().mockReturnValue(null),
		addTask: vi.fn(),
		updateTask: vi.fn(),
		getTask: vi.fn().mockReturnValue(undefined),
		addMessageToTaskHistory: vi.fn(),
		updateFromRunResult: vi.fn(),
	})),
}));

describe('ContextService', () => {
	let contextService: ContextService;
	let mockRepository: any;
	let mockContext: any;

	beforeEach(() => {
		vi.clearAllMocks();

		// Create mock context
		mockContext = {
			getTaskHistory: vi.fn().mockReturnValue([]),
			getTaskState: vi.fn().mockReturnValue(null),
			addTask: vi.fn(),
			updateTask: vi.fn(),
			getTask: vi.fn().mockReturnValue(undefined),
			addMessageToTaskHistory: vi.fn(),
			updateFromRunResult: vi.fn(),
		};

		// Create mock repository
		mockRepository = {
			list: vi.fn().mockResolvedValue([]),
			load: vi.fn().mockResolvedValue(mockContext),
			save: vi.fn().mockResolvedValue(undefined),
		};

		contextService = new ContextService(mockRepository);
	});

	describe('listContexts', () => {
		it('should return list of contexts from repository', async () => {
			const mockContexts = [mockContext, mockContext];
			mockRepository.list.mockResolvedValue(mockContexts);

			const result = await contextService.listContexts();

			expect(mockRepository.list).toHaveBeenCalled();
			expect(result).toEqual(mockContexts);
		});
	});

	describe('getContext', () => {
		it('should return context from repository', async () => {
			const result = await contextService.getContext('test-context-id');

			expect(mockRepository.load).toHaveBeenCalledWith('test-context-id');
			expect(result).toEqual(mockContext);
		});

		it('should return null when context not found', async () => {
			mockRepository.load.mockResolvedValue(null);

			const result = await contextService.getContext('non-existent-id');

			expect(result).toBeNull();
		});
	});

	describe('getOrCreate', () => {
		it('should throw error indicating not implemented', async () => {
			await expect(contextService.getOrCreate('test-id')).rejects.toThrow(
				'getOrCreate without agentId not yet implemented - contexts should be created by A2A server',
			);
		});
	});

	describe('updateFromRunResult', () => {
		it('should update context with run result', async () => {
			const mockRunResult = {history: []} as any;

			await contextService.updateFromRunResult(
				'test-context-id',
				'test-task-id',
				mockRunResult,
			);

			expect(mockRepository.load).toHaveBeenCalledWith('test-context-id');
			expect(mockContext.updateFromRunResult).toHaveBeenCalledWith(
				'test-task-id',
				mockRunResult,
			);
			expect(mockRepository.save).toHaveBeenCalledWith(mockContext);
		});

		it('should throw error when context not found', async () => {
			mockRepository.load.mockResolvedValue(null);
			const mockRunResult = {history: []} as any;

			await expect(
				contextService.updateFromRunResult(
					'non-existent-id',
					'test-task-id',
					mockRunResult,
				),
			).rejects.toThrow('Context non-existent-id not found');
		});
	});

	describe('getTaskHistory', () => {
		it('should return task history from context', async () => {
			const mockHistory: Message[] = [
				{
					messageId: 'msg-1',
					kind: 'message',
					role: 'user',
					parts: [{kind: 'text', text: 'Hello'}],
					contextId: 'test-context-id',
				},
			];
			mockContext.getTaskHistory.mockReturnValue(mockHistory);

			const result = await contextService.getTaskHistory(
				'test-context-id',
				'test-task-id',
			);

			expect(mockRepository.load).toHaveBeenCalledWith('test-context-id');
			expect(mockContext.getTaskHistory).toHaveBeenCalledWith('test-task-id');
			expect(result).toEqual(mockHistory);
		});

		it('should return empty array when context not found', async () => {
			mockRepository.load.mockResolvedValue(null);

			const result = await contextService.getTaskHistory(
				'non-existent-id',
				'test-task-id',
			);

			expect(result).toEqual([]);
		});
	});

	describe('getTaskState', () => {
		it('should return task state from context', async () => {
			const mockState = {status: 'active'};
			mockContext.getTaskState.mockReturnValue(mockState);

			const result = await contextService.getTaskState(
				'test-context-id',
				'test-task-id',
			);

			expect(mockRepository.load).toHaveBeenCalledWith('test-context-id');
			expect(mockContext.getTaskState).toHaveBeenCalledWith('test-task-id');
			expect(result).toEqual(mockState);
		});

		it('should return undefined when context not found', async () => {
			mockRepository.load.mockResolvedValue(null);

			const result = await contextService.getTaskState(
				'non-existent-id',
				'test-task-id',
			);

			expect(result).toBeUndefined();
		});
	});

	describe('addTask', () => {
		it('should add task to context', async () => {
			const mockTask: Task = {
				kind: 'task',
				id: 'test-task-id',
				contextId: 'test-context-id',
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};

			await contextService.addTask('test-context-id', mockTask);

			expect(mockRepository.load).toHaveBeenCalledWith('test-context-id');
			expect(mockContext.addTask).toHaveBeenCalledWith(mockTask);
			expect(mockRepository.save).toHaveBeenCalledWith(mockContext);
		});

		it('should throw error when context not found', async () => {
			mockRepository.load.mockResolvedValue(null);
			const mockTask: Task = {
				kind: 'task',
				id: 'test-task-id',
				contextId: 'test-context-id',
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};

			await expect(
				contextService.addTask('non-existent-id', mockTask),
			).rejects.toThrow('Context non-existent-id not found');
		});
	});

	describe('updateTask', () => {
		it('should update task in context', async () => {
			const mockTask: Task = {
				kind: 'task',
				id: 'test-task-id',
				contextId: 'test-context-id',
				status: {state: 'completed', timestamp: new Date().toISOString()},
				history: [],
			};

			await contextService.updateTask('test-context-id', mockTask);

			expect(mockRepository.load).toHaveBeenCalledWith('test-context-id');
			expect(mockContext.updateTask).toHaveBeenCalledWith(mockTask);
			expect(mockRepository.save).toHaveBeenCalledWith(mockContext);
		});

		it('should throw error when context not found', async () => {
			mockRepository.load.mockResolvedValue(null);
			const mockTask: Task = {
				kind: 'task',
				id: 'test-task-id',
				contextId: 'test-context-id',
				status: {state: 'completed', timestamp: new Date().toISOString()},
				history: [],
			};

			await expect(
				contextService.updateTask('non-existent-id', mockTask),
			).rejects.toThrow('Context non-existent-id not found');
		});
	});

	describe('getTask', () => {
		it('should return task from context', async () => {
			const mockTask: Task = {
				kind: 'task',
				id: 'test-task-id',
				contextId: 'test-context-id',
				status: {state: 'submitted', timestamp: new Date().toISOString()},
				history: [],
			};
			mockContext.getTask.mockReturnValue(mockTask);

			const result = await contextService.getTask(
				'test-context-id',
				'test-task-id',
			);

			expect(mockRepository.load).toHaveBeenCalledWith('test-context-id');
			expect(mockContext.getTask).toHaveBeenCalledWith('test-task-id');
			expect(result).toEqual(mockTask);
		});

		it('should return undefined when context not found', async () => {
			mockRepository.load.mockResolvedValue(null);

			const result = await contextService.getTask(
				'non-existent-id',
				'test-task-id',
			);

			expect(result).toBeUndefined();
		});
	});

	describe('addMessageToTaskHistory', () => {
		it('should add message to task history', async () => {
			const mockMessage: Message = {
				messageId: 'msg-1',
				kind: 'message',
				role: 'user',
				parts: [{kind: 'text', text: 'Hello'}],
				contextId: 'test-context-id',
			};

			await contextService.addMessageToTaskHistory(
				'test-context-id',
				'test-task-id',
				mockMessage,
			);

			expect(mockRepository.load).toHaveBeenCalledWith('test-context-id');
			expect(mockContext.addMessageToTaskHistory).toHaveBeenCalledWith(
				'test-task-id',
				mockMessage,
			);
			expect(mockRepository.save).toHaveBeenCalledWith(mockContext);
		});

		it('should throw error when context not found', async () => {
			mockRepository.load.mockResolvedValue(null);
			const mockMessage: Message = {
				messageId: 'msg-1',
				kind: 'message',
				role: 'user',
				parts: [{kind: 'text', text: 'Hello'}],
				contextId: 'test-context-id',
			};

			await expect(
				contextService.addMessageToTaskHistory(
					'non-existent-id',
					'test-task-id',
					mockMessage,
				),
			).rejects.toThrow('Context non-existent-id not found');
		});
	});

	describe('save', () => {
		it('should save context to repository', async () => {
			await contextService.save(mockContext);

			expect(mockRepository.save).toHaveBeenCalledWith(mockContext);
		});
	});
});
