import {describe, it, expect} from 'vitest';
import {
	TEST_TIMESTEP_PATHS,
	TEST_AGENTS,
	TEST_MODEL_PROVIDERS,
	TEST_MCP_SERVERS,
	TEST_CONTEXTS,
} from './testPaths.js';
import * as fs from 'node:fs';

describe('Test Fixtures', () => {
	it('should have valid fixture paths', () => {
		expect(TEST_TIMESTEP_PATHS.configDir).toContain('test-config');
		expect(TEST_TIMESTEP_PATHS.appConfig).toContain('app.json');
		expect(TEST_TIMESTEP_PATHS.agentsConfig).toContain('agents.jsonl');
		expect(TEST_TIMESTEP_PATHS.modelProviders).toContain(
			'modelProviders.jsonl',
		);
		expect(TEST_TIMESTEP_PATHS.mcpServers).toContain('mcpServers.jsonl');
		expect(TEST_TIMESTEP_PATHS.contexts).toContain('contexts.jsonl');
	});

	it('should have actual fixture files', () => {
		// Check that the fixture files actually exist
		expect(fs.existsSync(TEST_TIMESTEP_PATHS.appConfig)).toBe(true);
		expect(fs.existsSync(TEST_TIMESTEP_PATHS.agentsConfig)).toBe(true);
		expect(fs.existsSync(TEST_TIMESTEP_PATHS.modelProviders)).toBe(true);
		expect(fs.existsSync(TEST_TIMESTEP_PATHS.mcpServers)).toBe(true);
		expect(fs.existsSync(TEST_TIMESTEP_PATHS.contexts)).toBe(true);
	});

	it('should have valid JSON content in fixture files', () => {
		// Test app.json
		const appConfig = JSON.parse(
			fs.readFileSync(TEST_TIMESTEP_PATHS.appConfig, 'utf8'),
		);
		expect(appConfig.appPort).toBe(8080);
		expect(appConfig.environment).toBe('test');

		// Test agents.jsonl
		const agentsContent = fs.readFileSync(
			TEST_TIMESTEP_PATHS.agentsConfig,
			'utf8',
		);
		const agents = agentsContent
			.split('\n')
			.filter(line => line.trim())
			.map(line => JSON.parse(line));
		expect(agents).toHaveLength(2);
		expect(agents[0].id).toBe('test-agent-1');
		expect(agents[1].id).toBe('test-agent-2');

		// Test modelProviders.jsonl
		const providersContent = fs.readFileSync(
			TEST_TIMESTEP_PATHS.modelProviders,
			'utf8',
		);
		const providers = providersContent
			.split('\n')
			.filter(line => line.trim())
			.map(line => JSON.parse(line));
		expect(providers).toHaveLength(2);
		expect(providers[0].provider).toBe('openai');
		expect(providers[1].provider).toBe('ollama');
	});

	it('should have consistent test data', () => {
		expect(TEST_AGENTS).toHaveLength(2);
		expect(TEST_MODEL_PROVIDERS).toHaveLength(2);
		expect(TEST_MCP_SERVERS).toHaveLength(2);
		expect(TEST_CONTEXTS).toHaveLength(2);

		// Verify data consistency
		expect(TEST_AGENTS[0].modelProviderId).toBe('test-model-provider');
		expect(TEST_MODEL_PROVIDERS[0].id).toBe('test-model-provider');
	});
});
