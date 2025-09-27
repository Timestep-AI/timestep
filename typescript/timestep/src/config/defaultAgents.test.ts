import {describe, it, expect} from 'vitest';
import {getDefaultAgents} from './defaultAgents.js';

describe('defaultAgents', () => {
	describe('getDefaultAgents', () => {
		it('should return array of default agents', () => {
			const agents = getDefaultAgents();

			expect(Array.isArray(agents)).toBe(true);
			expect(agents.length).toBeGreaterThan(0);
		});

		it('should return agents with required properties', () => {
			const agents = getDefaultAgents();
			const firstAgent = agents[0];

			expect(firstAgent).toHaveProperty('id');
			expect(firstAgent).toHaveProperty('name');
			expect(firstAgent).toHaveProperty('instructions');
			expect(firstAgent).toHaveProperty('handoffIds');
			expect(firstAgent).toHaveProperty('toolIds');
			expect(firstAgent).toHaveProperty('model');
		});

		it('should have valid UUIDs for agent IDs', () => {
			const agents = getDefaultAgents();

			agents.forEach(agent => {
				expect(agent.id).toMatch(
					/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
				);
			});
		});

		it('should have valid UUIDs for handoff IDs', () => {
			const agents = getDefaultAgents();

			agents.forEach(agent => {
				if (agent.handoffIds) {
					agent.handoffIds.forEach(handoffId => {
						expect(handoffId).toMatch(
							/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
						);
					});
				}
			});
		});

		it('should have non-empty names and instructions', () => {
			const agents = getDefaultAgents();

			agents.forEach(agent => {
				expect(agent.name).toBeTruthy();
				expect(agent.name.length).toBeGreaterThan(0);
				expect(agent.instructions).toBeTruthy();
				expect(agent.instructions.length).toBeGreaterThan(0);
			});
		});
	});
});
