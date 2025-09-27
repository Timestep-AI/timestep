import {describe, it, expect} from 'vitest';
import type {Repository} from './repository.js';

describe('Repository interface', () => {
	it('should be importable', async () => {
		// Just test that we can import the repository module
		const repositoryModule = await import('./repository.js');
		expect(repositoryModule).toBeDefined();
	});

	it('should define Repository interface', () => {
		// Test that the Repository type is available
		const mockRepository: Repository<any, string> = {
			load: async () => null,
			save: async () => {},
			list: async () => [],
			delete: async () => {},
			exists: async () => false,
			getOrCreate: async () => ({} as any),
		};

		expect(mockRepository).toBeDefined();
		expect(typeof mockRepository.load).toBe('function');
		expect(typeof mockRepository.save).toBe('function');
		expect(typeof mockRepository.list).toBe('function');
		expect(typeof mockRepository.delete).toBe('function');
		expect(typeof mockRepository.exists).toBe('function');
		expect(typeof mockRepository.getOrCreate).toBe('function');
	});
});
