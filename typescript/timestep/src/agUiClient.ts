#!/usr/bin/env node

/**
 * AG-UI CLI Client
 *
 * Interactive command-line client for AG-UI (Agent User Interaction Protocol).
 * Provides real-time streaming chat with agents using Server-Sent Events.
 */

import readline from 'node:readline';
import crypto from 'node:crypto';
import process from 'node:process';
import {loadAppConfig} from './utils.js';

// Import official AG-UI SDK types
import {
	RunAgentInput,
	EventType,
	UserMessage,
} from '@ag-ui/core';

// --- AG-UI Types using official SDK ---
interface AGUIAgent {
	id: string;
	name: string;
	description: string;
	capabilities?: string[];
	status: string;
}

// --- ANSI Colors ---
const colors = {
	reset: '\x1b[0m',
	bright: '\x1b[1m',
	dim: '\x1b[2m',
	red: '\x1b[31m',
	green: '\x1b[32m',
	yellow: '\x1b[33m',
	blue: '\x1b[34m',
	magenta: '\x1b[35m',
	cyan: '\x1b[36m',
	gray: '\x1b[90m',
};

function colorize(color: keyof typeof colors, text: string): string {
	return `${colors[color]}${text}${colors.reset}`;
}

function generateId(): string {
	return crypto.randomUUID();
}

// --- State ---
let selectedAgentId: string | undefined = undefined;
let agentName = 'Agent';
const appConfig = loadAppConfig();
const baseServerUrl = `http://localhost:${appConfig.appPort}`;

// Command line args
interface CliArgs {
	agentId?: string;
	userInput?: string;
}

function parseCliArgs(): CliArgs {
	const args: CliArgs = {};
	for (let i = 2; i < process.argv.length; i++) {
		const arg = process.argv[i];
		if (arg === '--agentId' && i + 1 < process.argv.length) {
			args.agentId = process.argv[++i];
		} else if (arg === '--user-input' && i + 1 < process.argv.length) {
			args.userInput = process.argv[++i];
		}
	}

	return args;
}

const cliArgs = parseCliArgs();

// --- AG-UI Functions ---
async function discoverAgents(): Promise<AGUIAgent[]> {
	const response = await fetch(`${baseServerUrl}/ag-ui/agents/discover`);
	if (!response.ok)
		throw new Error(`Failed to discover agents: ${response.status}`);
	const data = await response.json();
	return data.agents || [];
}

async function selectAgent(): Promise<string> {
	const agents = await discoverAgents();
	if (agents.length === 0) throw new Error('No AG-UI agents available');

	if (cliArgs.agentId) {
		const agent = agents.find(a => a.id === cliArgs.agentId);
		if (!agent) {
			console.error(
				colorize('red', `❌ Agent ID ${cliArgs.agentId} not found`),
			);
			process.exit(1);
		}

		agentName = agent.name;
		return agent.id;
	}

	if (agents.length === 1) {
		agentName = agents[0].name;
		return agents[0].id;
	}

	console.log(colorize('bright', '\n🤖 Available AG-UI Agents:'));
	agents.forEach((agent, index) => {
		console.log(
			`${colorize('cyan', `${index + 1}.`)} ${colorize(
				'bright',
				agent.name,
			)} ${colorize('gray', `(${agent.id})`)}`,
		);
		console.log(`   ${colorize('dim', agent.description)}`);
		if (agent.capabilities && agent.capabilities.length > 0) {
			console.log(`   ${colorize('dim', `Capabilities: ${agent.capabilities.join(', ')}`)}`)
		}
	});

	return new Promise(resolve => {
		const askForSelection = () => {
			rl.question(
				colorize('cyan', `Enter your choice (1-${agents.length}): `),
				answer => {
					const choice = parseInt(answer.trim());
					if (isNaN(choice) || choice < 1 || choice > agents.length) {
						console.log(
							colorize(
								'red',
								`❌ Please enter a number between 1 and ${agents.length}`,
							),
						);
						askForSelection();
						return;
					}

					const selectedAgent = agents[choice - 1];
					agentName = selectedAgent.name;
					console.log(colorize('green', `✓ Selected: ${selectedAgent.name}\n`));
					resolve(selectedAgent.id);
				},
			);
		};

		askForSelection();
	});
}

async function runAgent(userInput: string): Promise<void> {
	if (!selectedAgentId) throw new Error('No agent selected');

	// Use official SDK RunAgentInput type
	const runInput: RunAgentInput = {
		runId: generateId(),
		threadId: 'cli-thread',
		state: {},
		tools: [],
		context: [],
		forwardedProps: {},
		messages: [
			{
				id: generateId(),
				role: 'user',
				content: userInput,
			} as UserMessage,
		],
	};

	const response = await fetch(
		`${baseServerUrl}/ag-ui/agents/${selectedAgentId}/run`,
		{
			method: 'POST',
			headers: {'Content-Type': 'application/json'},
			body: JSON.stringify(runInput),
		},
	);

	if (!response.ok) throw new Error(`AG-UI request failed: ${response.status}`);

	const reader = response.body?.getReader();
	if (!reader) throw new Error('No response body');

	const timestamp = new Date().toLocaleTimeString();
	const prefix = colorize('magenta', `${agentName} [${timestamp}]:`);

	try {
		while (true) {
			const {done, value} = await reader.read();
			if (done) break;

			const chunk = new TextDecoder().decode(value);
			const lines = chunk.split('\n').filter(line => line.trim());

			for (const line of lines) {
				if (line.startsWith('data: ')) {
					try {
						const event = JSON.parse(line.slice(6));

						switch (event.type) {
							case EventType.RUN_STARTED:
								console.log(`${prefix} ${colorize('blue', '⏳ Starting...')}`);
								break;
							case EventType.TEXT_MESSAGE_START:
								process.stdout.write(`${prefix} ${colorize('green', '')}`);
								break;
							case EventType.TEXT_MESSAGE_CONTENT:
								if (event.delta)
									process.stdout.write(colorize('green', event.delta));
								break;
							case EventType.TEXT_MESSAGE_END:
								console.log(); // New line
								break;
							case EventType.RUN_FINISHED:
								console.log(`${prefix} ${colorize('green', '✅ Completed')}`);
								break;
							case EventType.RUN_ERROR:
								console.log(
									`${prefix} ${colorize('red', `❌ Error: ${event.message}`)}`,
								);
								break;
						}
					} catch (e) {
						console.warn('Failed to parse event:', e);
					}
				}
			}
		}
	} finally {
		reader.releaseLock();
	}
}

// --- Readline Setup ---
const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
	prompt: colorize('cyan', 'You: '),
});

// --- Main CLI Logic ---
async function startCli(): Promise<void> {
	try {
		console.log(colorize('bright', '🚀 AG-UI CLI Client'));
		console.log(colorize('dim', `Connecting to: ${baseServerUrl}\n`));

		selectedAgentId = await selectAgent();

		if (cliArgs.userInput) {
			console.log(colorize('cyan', `You: ${cliArgs.userInput}`));
			await runAgent(cliArgs.userInput);
			process.exit(0);
		}

		console.log(colorize('bright', `💬 Chat with ${agentName}`));
		console.log(colorize('dim', 'Type your message or "quit" to exit.\n'));

		rl.prompt();

		rl.on('line', async (input: string) => {
			const trimmedInput = input.trim();

			if (trimmedInput === 'quit' || trimmedInput === 'exit') {
				console.log(colorize('dim', 'Goodbye! 👋'));
				rl.close();
				return;
			}

			if (trimmedInput === '') {
				rl.prompt();
				return;
			}

			try {
				await runAgent(trimmedInput);
			} catch (error) {
				console.error(
					colorize(
						'red',
						`❌ Error: ${error instanceof Error ? error.message : error}`,
					),
				);
			}

			rl.prompt();
		});

		rl.on('close', () => {
			console.log(colorize('dim', '\nGoodbye! 👋'));
			process.exit(0);
		});
	} catch (error) {
		console.error(
			colorize(
				'red',
				`❌ Failed to start CLI: ${
					error instanceof Error ? error.message : error
				}`,
			),
		);
		process.exit(1);
	}
}

// --- Start the CLI ---
if (import.meta.url === `file://${process.argv[1]}`) {
	startCli();
}
