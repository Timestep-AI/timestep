import {describe, it, expect} from 'vitest';
import {getDefaultModelProviders} from './defaultModelProviders.js';

describe('defaultModelProviders', () => {
	describe('getDefaultModelProviders', () => {
		it('should return array of default model providers', () => {
			const providers = getDefaultModelProviders();

			expect(Array.isArray(providers)).toBe(true);
			expect(providers.length).toBeGreaterThan(0);
		});

		it('should return providers with required properties', () => {
			const providers = getDefaultModelProviders();
			const firstProvider = providers[0];

			expect(firstProvider).toHaveProperty('id');
			expect(firstProvider).toHaveProperty('provider');
			expect(firstProvider).toHaveProperty('baseUrl');
			expect(firstProvider).toHaveProperty('modelsUrl');
		});

		it('should have valid UUIDs for provider IDs', () => {
			const providers = getDefaultModelProviders();

			providers.forEach(provider => {
				expect(provider.id).toMatch(
					/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
				);
			});
		});

		it('should have non-empty providers and URLs', () => {
			const providers = getDefaultModelProviders();

			providers.forEach(provider => {
				expect(provider.provider).toBeTruthy();
				expect(provider.provider.length).toBeGreaterThan(0);
				expect(provider.baseUrl).toBeTruthy();
				expect(provider.baseUrl.length).toBeGreaterThan(0);
				expect(provider.modelsUrl).toBeTruthy();
				expect(provider.modelsUrl.length).toBeGreaterThan(0);
			});
		});
	});
});
