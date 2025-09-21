/**
 * AG-UI (Agent User Interaction Protocol) Endpoints
 *
 * This module provides AG-UI endpoints for real-time frontend interaction with agents,
 * working alongside existing A2A endpoints without modifying them.
 */

import {Request, Response} from 'express';
import {TaskStore} from '@a2a-js/sdk/server';
import {AgentExecutor} from '@a2a-js/sdk/server';
import {AgentService} from '../services/agentService.js';
import {
	RepositoryContainer,
	DefaultRepositoryContainer,
} from '../services/backing/repositoryContainer.js';

// AG-UI Protocol Types
interface AGUIMessage {
	role: 'user' | 'assistant';
	parts: Array<{kind: 'text'; text: string}>;
	messageId: string;
}

interface AGUIRunInput {
	runId: string;
	threadId: string;
	messages: AGUIMessage[];
}

interface AGUIEvent {
	type: string;
	timestamp: string;
	runId?: string;
	content?: string;
	toolName?: string;
	error?: string;
}

interface AGUIAgent {
	id: string;
	name: string;
	description: string;
	skills: string[];
	status: string;
	handoffIds?: string[];
}

export class AGUIEndpoints {
	private agentService: AgentService;

	constructor(
		_taskStore: TaskStore,
		_agentExecutor: AgentExecutor,
		repositories: RepositoryContainer = new DefaultRepositoryContainer(),
	) {
		this.agentService = new AgentService(repositories.agents);
		// taskStore and agentExecutor are available if needed for future enhancements
	}

	// AG-UI Discovery Endpoint
	handleDiscover = async (_req: Request, res: Response): Promise<void> => {
		try {
			const agents = await this.agentService.listAgents();
			const aguiAgents: AGUIAgent[] = agents.map(agent => ({
				id: agent.id,
				name: agent.name,
				description:
					agent.handoffDescription ||
					agent.instructions.substring(0, 100) + '...',
				skills: this.extractSkills(agent.instructions),
				status: 'available',
				handoffIds: agent.handoffIds,
			}));

			res.json({agents: aguiAgents});
		} catch (error) {
			console.error('Error in AG-UI discover endpoint:', error);
			res.status(500).json({
				error: error instanceof Error ? error.message : 'Internal server error',
			});
		}
	};

	// AG-UI Run Endpoint (Server-Sent Events)
	handleRun = async (req: Request, res: Response): Promise<void> => {
		try {
			const runInput: AGUIRunInput = req.body;
			const {agentId} = req.params;

			if (!runInput.messages || !Array.isArray(runInput.messages)) {
				res.status(400).json({error: 'Invalid AG-UI input format'});
				return;
			}

			// Set up Server-Sent Events
			res.writeHead(200, {
				'Content-Type': 'text/event-stream',
				'Cache-Control': 'no-cache',
				Connection: 'keep-alive',
				'Access-Control-Allow-Origin': '*',
			});

			const sendEvent = (event: AGUIEvent) => {
				res.write(`data: ${JSON.stringify(event)}\n\n`);
			};

			// Send start event
			sendEvent({
				type: 'RUN_STARTED',
				timestamp: new Date().toISOString(),
				runId: runInput.runId,
			});

			try {
				// Get the user's message
				const userMessage = runInput.messages[runInput.messages.length - 1];
				const userText =
					userMessage.parts.find(p => p.kind === 'text')?.text || '';

				// Send message start
				sendEvent({
					type: 'TEXT_MESSAGE_START',
					timestamp: new Date().toISOString(),
				});

				// Simple echo response for now - replace with your actual agent logic
				const response = `Hello! You said: "${userText}". I'm agent ${agentId}.`;

				// Stream the response character by character
				for (let i = 0; i < response.length; i++) {
					sendEvent({
						type: 'TEXT_MESSAGE_CONTENT',
						timestamp: new Date().toISOString(),
						content: response[i],
					});
					// Small delay to demonstrate streaming
					await new Promise(resolve => setTimeout(resolve, 50));
				}

				// Send message end
				sendEvent({
					type: 'TEXT_MESSAGE_END',
					timestamp: new Date().toISOString(),
				});

				// Send completion
				sendEvent({
					type: 'RUN_FINISHED',
					timestamp: new Date().toISOString(),
					runId: runInput.runId,
				});
			} catch (error) {
				sendEvent({
					type: 'RUN_ERROR',
					timestamp: new Date().toISOString(),
					runId: runInput.runId,
					error: error instanceof Error ? error.message : 'Unknown error',
				});
			}

			res.end();
		} catch (error) {
			console.error('Error in AG-UI run endpoint:', error);
			res.status(500).json({
				error: error instanceof Error ? error.message : 'Internal server error',
			});
		}
	};

	private extractSkills(instructions: string): string[] {
		const skills: string[] = [];
		const lowerInstructions = instructions.toLowerCase();

		if (
			lowerInstructions.includes('code') ||
			lowerInstructions.includes('program')
		)
			skills.push('coding');
		if (
			lowerInstructions.includes('analyze') ||
			lowerInstructions.includes('data')
		)
			skills.push('analysis');
		if (
			lowerInstructions.includes('research') ||
			lowerInstructions.includes('search')
		)
			skills.push('research');
		if (
			lowerInstructions.includes('plan') ||
			lowerInstructions.includes('strategy')
		)
			skills.push('planning');

		skills.push('hello_world');
		return skills.length > 0 ? skills : ['general'];
	}
}

// Helper function to add AG-UI endpoints to existing Express app
export function addAGUIEndpoints(
	app: any,
	taskStore: TaskStore,
	agentExecutor: AgentExecutor,
	repositories?: RepositoryContainer,
): void {
	const aguiEndpoints = new AGUIEndpoints(
		taskStore,
		agentExecutor,
		repositories,
	);

	app.get('/ag-ui/agents/discover', aguiEndpoints.handleDiscover);
	app.post('/ag-ui/agents/:agentId/run', aguiEndpoints.handleRun);

	console.log('🎯 AG-UI endpoints added:');
	console.log('   GET  /ag-ui/agents/discover');
	console.log('   POST /ag-ui/agents/:agentId/run');
}
