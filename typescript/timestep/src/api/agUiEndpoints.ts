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

// AG-UI Protocol Types - Based on official specification
interface AGUIMessage {
	role: 'user' | 'assistant';
	content: string;
	messageId?: string;
}

interface AGUIRunInput {
	runId: string;
	threadId: string;
	messages: AGUIMessage[];
}

// Official AG-UI Event Types
enum EventType {
	RUN_STARTED = 'RUN_STARTED',
	TEXT_MESSAGE_START = 'TEXT_MESSAGE_START',
	TEXT_MESSAGE_CONTENT = 'TEXT_MESSAGE_CONTENT',
	TEXT_MESSAGE_END = 'TEXT_MESSAGE_END',
	RUN_FINISHED = 'RUN_FINISHED',
	RUN_ERROR = 'RUN_ERROR',
}

interface BaseEvent {
	type: EventType;
	threadId?: string;
	runId?: string;
}

interface RunStartedEvent extends BaseEvent {
	type: EventType.RUN_STARTED;
	threadId: string;
	runId: string;
}

interface TextMessageStartEvent extends BaseEvent {
	type: EventType.TEXT_MESSAGE_START;
	messageId: string;
	role: 'assistant';
}

interface TextMessageContentEvent extends BaseEvent {
	type: EventType.TEXT_MESSAGE_CONTENT;
	messageId: string;
	delta: string;
}

interface TextMessageEndEvent extends BaseEvent {
	type: EventType.TEXT_MESSAGE_END;
	messageId: string;
}

interface RunFinishedEvent extends BaseEvent {
	type: EventType.RUN_FINISHED;
	threadId: string;
	runId: string;
}

interface RunErrorEvent extends BaseEvent {
	type: EventType.RUN_ERROR;
	threadId: string;
	runId: string;
	error: string;
}

type AGUIEvent = RunStartedEvent | TextMessageStartEvent | TextMessageContentEvent | TextMessageEndEvent | RunFinishedEvent | RunErrorEvent;

interface AGUIAgent {
	id: string;
	name: string;
	description: string;
	status: string;
	capabilities?: string[];
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
				description: agent.handoffDescription || agent.instructions.substring(0, 100) + '...',
				status: 'available',
				capabilities: this.extractCapabilities(agent.instructions),
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

			// Send RUN_STARTED event
			const runStartedEvent: RunStartedEvent = {
				type: EventType.RUN_STARTED,
				threadId: runInput.threadId,
				runId: runInput.runId,
			};
			sendEvent(runStartedEvent);

			try {
				// Get the user's message
				const userMessage = runInput.messages[runInput.messages.length - 1];
				const userText = userMessage.content || '';

				// Generate message ID
				const messageId = `msg_${Date.now()}`;

				// Send TEXT_MESSAGE_START event
				const messageStartEvent: TextMessageStartEvent = {
					type: EventType.TEXT_MESSAGE_START,
					messageId,
					role: 'assistant',
				};
				sendEvent(messageStartEvent);

				// Generate response
				const response = `Hello! You said: "${userText}". I'm agent ${agentId}.`;

				// Stream the response character by character
				for (let i = 0; i < response.length; i++) {
					const contentEvent: TextMessageContentEvent = {
						type: EventType.TEXT_MESSAGE_CONTENT,
						messageId,
						delta: response[i],
					};
					sendEvent(contentEvent);
					// Small delay to demonstrate streaming
					await new Promise(resolve => setTimeout(resolve, 50));
				}

				// Send TEXT_MESSAGE_END event
				const messageEndEvent: TextMessageEndEvent = {
					type: EventType.TEXT_MESSAGE_END,
					messageId,
				};
				sendEvent(messageEndEvent);

				// Send RUN_FINISHED event
				const runFinishedEvent: RunFinishedEvent = {
					type: EventType.RUN_FINISHED,
					threadId: runInput.threadId,
					runId: runInput.runId,
				};
				sendEvent(runFinishedEvent);
			} catch (error) {
				const runErrorEvent: RunErrorEvent = {
					type: EventType.RUN_ERROR,
					threadId: runInput.threadId,
					runId: runInput.runId,
					error: error instanceof Error ? error.message : 'Unknown error',
				};
				sendEvent(runErrorEvent);
			}

			res.end();
		} catch (error) {
			console.error('Error in AG-UI run endpoint:', error);
			res.status(500).json({
				error: error instanceof Error ? error.message : 'Internal server error',
			});
		}
	};

	private extractCapabilities(instructions: string): string[] {
		const capabilities: string[] = [];
		const lowerInstructions = instructions.toLowerCase();

		if (
			lowerInstructions.includes('code') ||
			lowerInstructions.includes('program')
		)
			capabilities.push('coding');
		if (
			lowerInstructions.includes('analyze') ||
			lowerInstructions.includes('data')
		)
			capabilities.push('analysis');
		if (
			lowerInstructions.includes('research') ||
			lowerInstructions.includes('search')
		)
			capabilities.push('research');
		if (
			lowerInstructions.includes('plan') ||
			lowerInstructions.includes('strategy')
		)
			capabilities.push('planning');

		capabilities.push('conversation');
		return capabilities.length > 0 ? capabilities : ['general'];
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
