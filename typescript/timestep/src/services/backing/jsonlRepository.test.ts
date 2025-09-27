import {describe, it, expect, beforeEach} from 'vitest';
import {JsonlRepository} from './jsonlRepository.js';

// Create a concrete implementation for testing
class TestJsonlRepository extends JsonlRepository<
	{id: string; name: string},
	string
> {
	constructor() {
		super('/tmp/test.jsonl');
	}

	protected serialize(entity: {id: string; name: string}): string {
		return JSON.stringify(entity);
	}

	protected deserialize(line: string): {id: string; name: string} {
		return JSON.parse(line);
	}

	protected getId(entity: {id: string; name: string}): string {
		return entity.id;
	}
}

describe('JsonlRepository', () => {
	let repository: TestJsonlRepository;

	beforeEach(() => {
		repository = new TestJsonlRepository();
	});

	describe('constructor', () => {
		it('should initialize with file path', () => {
			expect(repository).toBeDefined();
		});
	});

	describe('serialize', () => {
		it('should serialize entity to JSON', () => {
			const entity = {id: 'test-1', name: 'Test Entity'};
			const result = (repository as any).serialize(entity);
			expect(result).toBe('{"id":"test-1","name":"Test Entity"}');
		});
	});

	describe('deserialize', () => {
		it('should deserialize JSON to entity', () => {
			const json = '{"id":"test-1","name":"Test Entity"}';
			const result = (repository as any).deserialize(json);
			expect(result).toEqual({id: 'test-1', name: 'Test Entity'});
		});
	});

	describe('getId', () => {
		it('should extract ID from entity', () => {
			const entity = {id: 'test-1', name: 'Test Entity'};
			const result = (repository as any).getId(entity);
			expect(result).toBe('test-1');
		});
	});
});
