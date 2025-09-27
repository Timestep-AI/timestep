import {describe, it, expect} from 'vitest';
import {listTraces} from './tracesApi.js';

describe('tracesApi', () => {
	describe('listTraces', () => {
		it('should be a function', () => {
			expect(typeof listTraces).toBe('function');
		});

		it('should return a promise', () => {
			const result = listTraces();
			expect(result).toBeInstanceOf(Promise);
		});
	});
});
