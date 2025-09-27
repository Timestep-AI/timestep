import {describe, it, expect} from 'vitest';
import {DefaultRepositoryContainer} from './repositoryContainer.js';

describe('DefaultRepositoryContainer', () => {
	describe('constructor', () => {
		it('should create instance with all repositories', () => {
			const container = new DefaultRepositoryContainer();

			expect(container).toHaveProperty('agents');
			expect(container).toHaveProperty('contexts');
			expect(container).toHaveProperty('modelProviders');
			expect(container).toHaveProperty('mcpServers');
		});
	});
});
