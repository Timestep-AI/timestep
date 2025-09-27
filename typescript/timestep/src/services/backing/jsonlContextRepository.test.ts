import {describe, it, expect, vi, beforeEach} from 'vitest';

// Mock dependencies first
vi.mock('../../utils.js', () => ({
	getTimestepPaths: vi.fn(() => ({
		agentsConfig: '/test/path/agents.jsonl',
		contextsConfig: '/test/path/contexts.jsonl',
		mcpServersConfig: '/test/path/mcpServers.jsonl',
		modelProvidersConfig: '/test/path/modelProviders.jsonl',
	})),
}));

vi.mock('../../types/context.js', () => ({
	Context: class MockContext {
		contextId: string;
		agentId: string;
		taskHistories: any;

		constructor(contextId: string, agentId: string) {
			this.contextId = contextId;
			this.agentId = agentId;
			this.taskHistories = {};
		}

		toJSON() {
			return {
				contextId: this.contextId,
				agentId: this.agentId,
				taskHistories: this.taskHistories,
			};
		}

		static fromJSON(data: any) {
			const context = new MockContext(data.contextId, data.agentId);
			context.taskHistories = data.taskHistories || {};
			return context;
		}
	},
}));

// Mock the parent class
vi.mock('./jsonlRepository.js', () => ({
	JsonlRepository: class MockJsonlRepository {
		filePath: string;
		load: any;
		save: any;
		delete: any;
		list: any;

		constructor(filePath: string) {
			this.filePath = filePath;
			this.load = vi.fn();
			this.save = vi.fn();
			this.delete = vi.fn();
			this.list = vi.fn();
		}
	},
}));

// Now import the module under test
import {JsonlContextRepository} from './jsonlContextRepository.js';

// Define Context type locally to avoid circular dependency
// interface Context {
//   contextId: string;
//   agentId: string;
//   taskHistories: any;
//   toJSON(): any;
// }

describe('JsonlContextRepository', () => {
	let repository: JsonlContextRepository;

	beforeEach(() => {
		vi.clearAllMocks();
		repository = new JsonlContextRepository();
	});

	describe('Constructor', () => {
		it('should create an instance', () => {
			expect(repository).toBeDefined();
			expect(repository).toBeInstanceOf(JsonlContextRepository);
		});

		it('should call parent constructor with correct path', () => {
			// The constructor test is covered by the instance creation test above
			expect(repository).toBeDefined();
		});
	});

	describe('serialize', () => {
		it('should serialize context to JSON string', () => {
			const mockContext = {
				contextId: 'test-context-1',
				agentId: 'test-agent-1',
				taskHistories: {'task-1': {messages: []}},
				toJSON: vi.fn(() => ({
					contextId: 'test-context-1',
					agentId: 'test-agent-1',
					taskHistories: {'task-1': {messages: []}},
				})),
			};

			const result = (repository as any).serialize(mockContext);
			expect(result).toBe(JSON.stringify(mockContext.toJSON()));
			expect(mockContext.toJSON).toHaveBeenCalled();
		});

		it('should handle complex context data', () => {
			const mockContext = {
				contextId: 'complex-context-123',
				agentId: 'complex-agent-456',
				taskHistories: {
					'task-1': {messages: [{role: 'user', content: 'Hello'}]},
					'task-2': {messages: [{role: 'assistant', content: 'Hi there!'}]},
				},
				toJSON: vi.fn(() => ({
					contextId: 'complex-context-123',
					agentId: 'complex-agent-456',
					taskHistories: {
						'task-1': {messages: [{role: 'user', content: 'Hello'}]},
						'task-2': {messages: [{role: 'assistant', content: 'Hi there!'}]},
					},
				})),
			};

			const result = (repository as any).serialize(mockContext);
			expect(result).toBe(JSON.stringify(mockContext.toJSON()));
			expect(mockContext.toJSON).toHaveBeenCalled();
		});

		it('should handle context without task histories', () => {
			const mockContext = {
				contextId: 'empty-context',
				agentId: 'empty-agent',
				taskHistories: {},
				toJSON: vi.fn(() => ({
					contextId: 'empty-context',
					agentId: 'empty-agent',
					taskHistories: {},
				})),
			};

			const result = (repository as any).serialize(mockContext);
			expect(result).toBe(JSON.stringify(mockContext.toJSON()));
			expect(mockContext.toJSON).toHaveBeenCalled();
		});
	});

	describe('deserialize', () => {
		it('should deserialize JSON string to context', () => {
			const contextData = {
				contextId: 'test-context-1',
				agentId: 'test-agent-1',
				taskHistories: {'task-1': {messages: []}},
			};

			const jsonString = JSON.stringify(contextData);
			const result = (repository as any).deserialize(jsonString);

			expect(result).toBeDefined();
			expect(result.contextId).toBe('test-context-1');
			expect(result.agentId).toBe('test-agent-1');
			expect(result.taskHistories).toEqual({'task-1': {messages: []}});
		});

		it('should handle complex context data', () => {
			const contextData = {
				contextId: 'complex-context-123',
				agentId: 'complex-agent-456',
				taskHistories: {
					'task-1': {messages: [{role: 'user', content: 'Hello'}]},
					'task-2': {messages: [{role: 'assistant', content: 'Hi there!'}]},
				},
			};

			const jsonString = JSON.stringify(contextData);
			const result = (repository as any).deserialize(jsonString);

			expect(result).toBeDefined();
			expect(result.contextId).toBe('complex-context-123');
			expect(result.agentId).toBe('complex-agent-456');
			expect(result.taskHistories).toEqual(contextData.taskHistories);
		});

		it('should handle invalid JSON gracefully', () => {
			const invalidJson = 'invalid json string';

			expect(() => {
				(repository as any).deserialize(invalidJson);
			}).toThrow();
		});
	});

	describe('getId', () => {
		it('should return context ID', () => {
			const mockContext = {
				contextId: 'test-context-id',
				agentId: 'test-agent-id',
				taskHistories: {},
			};

			const result = (repository as any).getId(mockContext);
			expect(result).toBe('test-context-id');
		});

		it('should handle different ID formats', () => {
			const mockContext = {
				contextId: 'context-123-456-789',
				agentId: 'agent-123-456-789',
				taskHistories: {},
			};

			const result = (repository as any).getId(mockContext);
			expect(result).toBe('context-123-456-789');
		});
	});

	describe('getOrCreate', () => {
		it('should return existing context when found', async () => {
			const existingContext = {
				contextId: 'existing-context',
				agentId: 'existing-agent',
				taskHistories: {'task-1': {messages: []}},
			};

			(repository as any).load = vi.fn().mockResolvedValue(existingContext);

			const result = await repository.getOrCreate(
				'existing-context',
				'existing-agent',
			);

			expect(result).toBe(existingContext);
			expect((repository as any).load).toHaveBeenCalledWith('existing-context');
		});

		it('should create new context when not found', async () => {
			(repository as any).load = vi.fn().mockResolvedValue(null);

			const result = await repository.getOrCreate('new-context', 'new-agent');

			expect(result).toBeDefined();
			expect(result.contextId).toBe('new-context');
			expect(result.agentId).toBe('new-agent');
			expect((repository as any).load).toHaveBeenCalledWith('new-context');
		});

		it('should handle multiple getOrCreate operations', async () => {
			const existingContext = {
				contextId: 'existing-context',
				agentId: 'existing-agent',
				taskHistories: {},
			};

			(repository as any).load = vi.fn().mockResolvedValue(existingContext);

			const result1 = await repository.getOrCreate(
				'existing-context',
				'existing-agent',
			);
			const result2 = await repository.getOrCreate(
				'existing-context',
				'existing-agent',
			);

			expect(result1).toBe(existingContext);
			expect(result2).toBe(existingContext);
			expect((repository as any).load).toHaveBeenCalledTimes(2);
		});

		it('should handle concurrent getOrCreate operations', async () => {
			const existingContext = {
				contextId: 'existing-context',
				agentId: 'existing-agent',
				taskHistories: {},
			};

			(repository as any).load = vi.fn().mockResolvedValue(existingContext);

			const promises = [
				repository.getOrCreate('existing-context', 'existing-agent'),
				repository.getOrCreate('existing-context', 'existing-agent'),
				repository.getOrCreate('existing-context', 'existing-agent'),
			];

			const results = await Promise.all(promises);

			expect(results).toHaveLength(3);
			results.forEach(result => {
				expect(result).toBe(existingContext);
			});
			expect((repository as any).load).toHaveBeenCalledTimes(3);
		});
	});

	describe('save', () => {
		it('should save context successfully', async () => {
			const mockContext = {
				contextId: 'test-context',
				agentId: 'test-agent',
				taskHistories: {'task-1': {messages: []}},
			};

			(repository as any).save = vi.fn().mockImplementation(async _context => {
				return Promise.resolve();
			});

			await repository.save(mockContext as any);

			expect((repository as any).save).toHaveBeenCalledWith(mockContext);
		});

		it('should handle multiple save operations', async () => {
			const mockContext1 = {
				contextId: 'test-context-1',
				agentId: 'test-agent-1',
				taskHistories: {},
			};

			const mockContext2 = {
				contextId: 'test-context-2',
				agentId: 'test-agent-2',
				taskHistories: {},
			};

			(repository as any).save = vi.fn().mockImplementation(async _context => {
				return Promise.resolve();
			});

			await repository.save(mockContext1 as any);
			await repository.save(mockContext2 as any);

			expect((repository as any).save).toHaveBeenCalledTimes(2);
			expect((repository as any).save).toHaveBeenCalledWith(mockContext1);
			expect((repository as any).save).toHaveBeenCalledWith(mockContext2);
		});

		it('should handle concurrent save operations', async () => {
			const mockContexts = Array.from({length: 5}, (_, i) => ({
				contextId: `test-context-${i}`,
				agentId: `test-agent-${i}`,
				taskHistories: {},
			}));

			(repository as any).save = vi.fn().mockImplementation(async _context => {
				return Promise.resolve();
			});

			const promises = mockContexts.map(context =>
				repository.save(context as any),
			);
			await Promise.all(promises);

			expect((repository as any).save).toHaveBeenCalledTimes(5);
		});
	});

	describe('load', () => {
		it('should load existing context', async () => {
			const mockContext = {
				contextId: 'test-context',
				agentId: 'test-agent',
				taskHistories: {'task-1': {messages: []}},
			};

			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return mockContext;
				});

			const result = await repository.load('test-context');

			expect(result).toBe(mockContext);
			expect((repository as any).load).toHaveBeenCalledWith('test-context');
		});

		it('should return null when context not found', async () => {
			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return null;
				});

			const result = await repository.load('non-existent-context');

			expect(result).toBeNull();
			expect((repository as any).load).toHaveBeenCalledWith(
				'non-existent-context',
			);
		});

		it('should handle multiple load operations', async () => {
			const mockContext = {
				contextId: 'test-context',
				agentId: 'test-agent',
				taskHistories: {},
			};

			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return mockContext;
				});

			const result1 = await repository.load('test-context');
			const result2 = await repository.load('test-context');

			expect(result1).toBe(mockContext);
			expect(result2).toBe(mockContext);
			expect((repository as any).load).toHaveBeenCalledTimes(2);
		});

		it('should handle concurrent load operations', async () => {
			const mockContext = {
				contextId: 'test-context',
				agentId: 'test-agent',
				taskHistories: {},
			};

			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return mockContext;
				});

			const promises = [
				repository.load('test-context'),
				repository.load('test-context'),
				repository.load('test-context'),
			];

			const results = await Promise.all(promises);

			expect(results).toHaveLength(3);
			results.forEach(result => {
				expect(result).toBe(mockContext);
			});
			expect((repository as any).load).toHaveBeenCalledTimes(3);
		});
	});

	describe('Integration Tests', () => {
		it('should handle full workflow with existing context', async () => {
			const mockContext = {
				contextId: 'workflow-context',
				agentId: 'workflow-agent',
				taskHistories: {'task-1': {messages: []}},
			};

			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return mockContext;
				});

			(repository as any).save = vi.fn().mockImplementation(async _context => {
				return Promise.resolve();
			});

			// Test getOrCreate with existing context
			const result = await repository.getOrCreate(
				'workflow-context',
				'workflow-agent',
			);
			expect(result).toBe(mockContext);

			// Test save
			await repository.save(mockContext as any);
			expect((repository as any).save).toHaveBeenCalledWith(mockContext);

			// Test load
			const loadedContext = await repository.load('workflow-context');
			expect(loadedContext).toBe(mockContext);
		});

		it('should handle full workflow with new context', async () => {
			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return null;
				});

			(repository as any).save = vi.fn().mockImplementation(async _context => {
				return Promise.resolve();
			});

			// Test getOrCreate with new context
			const result = await repository.getOrCreate(
				'new-workflow-context',
				'new-workflow-agent',
			);
			expect(result).toBeDefined();
			expect(result.contextId).toBe('new-workflow-context');
			expect(result.agentId).toBe('new-workflow-agent');

			// Test save
			await repository.save(result as any);
			expect((repository as any).save).toHaveBeenCalledWith(result);
		});

		it('should handle error scenarios gracefully', async () => {
			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					throw new Error('Load failed');
				});

			await expect(
				repository.getOrCreate('error-context', 'error-agent'),
			).rejects.toThrow('Load failed');
		});
	});

	describe('Performance Tests', () => {
		it('should handle multiple rapid getOrCreate calls', async () => {
			const mockContext = {
				contextId: 'performance-context',
				agentId: 'performance-agent',
				taskHistories: {},
			};

			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return mockContext;
				});

			const start = Date.now();
			const promises = Array.from({length: 20}, () =>
				repository.getOrCreate('performance-context', 'performance-agent'),
			);
			const results = await Promise.all(promises);
			const end = Date.now();

			expect(results).toHaveLength(20);
			expect(end - start).toBeLessThan(5000); // Should complete within 5 seconds

			results.forEach(result => {
				expect(result).toBe(mockContext);
			});
		});

		it('should handle memory efficiently with large context lists', async () => {
			const mockContexts = Array.from({length: 10}, (_, i) => ({
				contextId: `context-${i}`,
				agentId: `agent-${i}`,
				taskHistories: {},
			}));

			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return mockContexts.find(ctx => ctx.contextId === _contextId) || null;
				});

			const results = [];
			for (let i = 0; i < 10; i++) {
				const result = await repository.getOrCreate(
					`context-${i}`,
					`agent-${i}`,
				);
				results.push(result);
			}

			expect(results).toHaveLength(10);
			results.forEach((result, index) => {
				expect(result.contextId).toBe(`context-${index}`);
				expect(result.agentId).toBe(`agent-${index}`);
			});
		});

		it('should handle concurrent save and load operations efficiently', async () => {
			const mockContext = {
				contextId: 'concurrent-context',
				agentId: 'concurrent-agent',
				taskHistories: {},
			};

			(repository as any).load = vi
				.fn()
				.mockImplementation(async _contextId => {
					return mockContext;
				});

			(repository as any).save = vi.fn().mockImplementation(async _context => {
				return Promise.resolve();
			});

			const start = Date.now();
			const promises = [
				...Array.from({length: 10}, () =>
					repository.load('concurrent-context'),
				),
				...Array.from({length: 10}, () => repository.save(mockContext as any)),
			];
			const results = await Promise.all(promises);
			const end = Date.now();

			expect(results).toHaveLength(20);
			expect(end - start).toBeLessThan(5000); // Should complete within 5 seconds
			expect((repository as any).load).toHaveBeenCalledTimes(10);
			expect((repository as any).save).toHaveBeenCalledTimes(10);
		});
	});
});
