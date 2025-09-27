import {describe, it, expect, vi, beforeEach} from 'vitest';
import {AgentService} from './agentService.js';
import {Agent} from '../api/agentsApi.js';

describe('AgentService', () => {
	let agentService: AgentService;
	let mockRepository: any;

	beforeEach(() => {
		vi.clearAllMocks();

		mockRepository = {
			list: vi.fn(),
			load: vi.fn(),
			exists: vi.fn(),
			save: vi.fn(),
			delete: vi.fn(),
		};

		agentService = new AgentService(mockRepository);
	});

	describe('listAgents', () => {
		it('should return list of agents from repository', async () => {
			const mockAgents: Agent[] = [
				{
					id: 'agent-1',
					name: 'Agent 1',
					instructions: 'Test instructions',
					model: 'provider-1',
					modelSettings: {temperature: 0.7},
					toolIds: [],
				},
				{
					id: 'agent-2',
					name: 'Agent 2',
					instructions: 'Test instructions 2',
					model: 'provider-1',
					modelSettings: {temperature: 0.7},
					toolIds: ['tool-1'],
				},
			];
			mockRepository.list.mockResolvedValue(mockAgents);

			const result = await agentService.listAgents();

			expect(mockRepository.list).toHaveBeenCalled();
			expect(result).toEqual(mockAgents);
		});
	});

	describe('getAgent', () => {
		it('should return agent from repository', async () => {
			const mockAgent: Agent = {
				id: 'agent-1',
				name: 'Agent 1',
				instructions: 'Test instructions',
				model: 'provider-1',
				modelSettings: {temperature: 0.7},
				toolIds: [],
			};
			mockRepository.load.mockResolvedValue(mockAgent);

			const result = await agentService.getAgent('agent-1');

			expect(mockRepository.load).toHaveBeenCalledWith('agent-1');
			expect(result).toEqual(mockAgent);
		});

		it('should return null when agent not found', async () => {
			mockRepository.load.mockResolvedValue(null);

			const result = await agentService.getAgent('non-existent-agent');

			expect(result).toBeNull();
		});
	});

	describe('isAgentAvailable', () => {
		it('should return true when agent exists', async () => {
			mockRepository.exists.mockResolvedValue(true);

			const result = await agentService.isAgentAvailable('agent-1');

			expect(mockRepository.exists).toHaveBeenCalledWith('agent-1');
			expect(result).toBe(true);
		});

		it('should return false when agent does not exist', async () => {
			mockRepository.exists.mockResolvedValue(false);

			const result = await agentService.isAgentAvailable('non-existent-agent');

			expect(result).toBe(false);
		});
	});

	describe('saveAgent', () => {
		it('should save agent to repository', async () => {
			const mockAgent: Agent = {
				id: 'agent-1',
				name: 'Agent 1',
				instructions: 'Test instructions',
				model: 'provider-1',
				modelSettings: {temperature: 0.7},
				toolIds: [],
			};

			await agentService.saveAgent(mockAgent);

			expect(mockRepository.save).toHaveBeenCalledWith(mockAgent);
		});
	});

	describe('deleteAgent', () => {
		it('should delete agent from repository', async () => {
			await agentService.deleteAgent('agent-1');

			expect(mockRepository.delete).toHaveBeenCalledWith('agent-1');
		});
	});
});
