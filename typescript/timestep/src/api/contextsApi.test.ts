import {describe, it, expect, vi, beforeEach} from 'vitest';
import {listContexts, getContext} from './contextsApi.js';

// Mock the ContextService
const mockContextService = {
	listContexts: vi.fn(),
	getContext: vi.fn(),
};

vi.mock('../services/contextService.js', () => ({
	ContextService: vi.fn().mockImplementation(() => mockContextService),
}));

// Mock the repository container
vi.mock('../services/backing/repositoryContainer.js', () => ({
	DefaultRepositoryContainer: vi.fn().mockImplementation(() => ({
		contexts: {
			list: vi.fn(),
			load: vi.fn(),
		},
	})),
}));

describe('contextsApi', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	describe('listContexts', () => {
		it('should be a function', () => {
			expect(typeof listContexts).toBe('function');
		});

		it('should return a promise', () => {
			const result = listContexts();
			expect(result).toBeInstanceOf(Promise);
		});

		it('should handle successful context listing', async () => {
			mockContextService.listContexts.mockResolvedValue([
				{id: 'context-1', tasks: []},
				{id: 'context-2', tasks: []},
			]);

			const result = await listContexts();

			expect(result).toEqual({
				data: [
					{id: 'context-1', tasks: []},
					{id: 'context-2', tasks: []},
				],
			});
		});

		it('should handle error in context listing', async () => {
			mockContextService.listContexts.mockRejectedValue(
				new Error('Database error'),
			);

			await expect(listContexts()).rejects.toThrow(
				'Failed to read contexts: Error: Database error',
			);
		});
	});

	describe('getContext', () => {
		it('should be a function', () => {
			expect(typeof getContext).toBe('function');
		});

		it('should return a promise', () => {
			const result = getContext('test-id');
			expect(result).toBeInstanceOf(Promise);
		});

		it('should handle successful context retrieval', async () => {
			mockContextService.getContext.mockResolvedValue({
				id: 'context-1',
				tasks: [],
			});

			const result = await getContext('context-1');

			expect(result).toEqual({id: 'context-1', tasks: []});
		});

		it('should handle context not found', async () => {
			mockContextService.getContext.mockResolvedValue(null);

			const result = await getContext('non-existent-context');

			expect(result).toBeNull();
		});

		it('should handle error in context retrieval', async () => {
			mockContextService.getContext.mockRejectedValue(
				new Error('Database error'),
			);

			await expect(getContext('context-1')).rejects.toThrow(
				'Failed to get context: Error: Database error',
			);
		});
	});
});
