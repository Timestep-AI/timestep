#!/usr/bin/env node

import readline from 'node:readline';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
// import { fileURLToPath } from "node:url";
// import { spawn } from "node:child_process";
// import { Client } from "@modelcontextprotocol/sdk/client/index.js";
// import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
// import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
// import { CallToolRequestSchema, CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";

import {
	// Specific Params/Payload types used by the CLI
	MessageSendParams, // Changed from TaskSendParams
	TaskStatusUpdateEvent,
	TaskArtifactUpdateEvent,
	Message,
	Task, // Added for direct Task events
	// Other types needed for message/part handling
	// TaskState,
	FilePart,
	DataPart,
	// Type for the agent card
	AgentCard,
	Part, // Added for explicit Part typing
} from '@a2a-js/sdk';

import {A2AClient} from '@a2a-js/sdk/client';
import {getTimestepPaths, loadAppConfig} from './utils.js';

// --- Types ---
interface ToolCall {
	id: string;
	name: string;
	parameters: Record<string, any>;
	artifactId: string;
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

// --- Helper Functions ---
function colorize(color: keyof typeof colors, text: string): string {
	return `${colors[color]}${text}${colors.reset}`;
}

function generateId(): string {
	// Renamed for more general use
	return crypto.randomUUID();
}

// Tool event types for clean display
interface ToolCallEvent {
	type: 'tool_call';
	name: string;
	arguments: string;
	agent: string;
	taskId: string;
}

interface ToolResponseEvent {
	type: 'tool_response';
	name: string;
	input: string;
	output: string;
	status: 'success' | 'error';
	agent: string;
	taskId: string;
}

type ToolEvent = ToolCallEvent | ToolResponseEvent;

function checkForToolEvent(message: Message): ToolEvent | null {
	if (!message.parts) return null;

	for (const part of message.parts) {
		if (part.kind === 'data') {
			const data = (part as DataPart).data as Record<string, unknown>;

			// Skip handoff events - they should be displayed as Assistant messages, not Tool boxes
			const eventType = data?.['type'] as string;
			if (
				eventType === 'handoff_requested' ||
				eventType === 'handoff_occurred'
			) {
				return null;
			}

			// Check for tool call
			if (data?.['name'] && data?.['arguments']) {
				const toolName = data['name'] as string;
				const toolArgs = data['arguments'] as string;

				// Skip handoff function calls - they should be handled as Assistant messages only
				if (toolName.startsWith('transfer_to_')) {
					return null;
				}

				// Store the tool call inputs for later use in responses
				toolCallInputs.set(toolName, toolArgs);

				return {
					type: 'tool_call',
					name: toolName,
					arguments: toolArgs,
					agent: agentName,
					taskId: currentTaskId?.substring(0, 8) || 'unknown',
				};
			}

			// Check for tool output/response
			if (
				data?.['type'] === 'function_call_result' &&
				data?.['name'] &&
				data?.['output']
			) {
				const toolName = data['name'] as string;

				// Show handoff function results as Tool messages
				// (they were previously suppressed but should be displayed)

				const output = data?.['output'] as Record<string, unknown>;
				const outputText = output?.['text'] || JSON.stringify(output);

				// Get the stored input arguments from when the tool was called
				const storedInput = toolCallInputs.get(toolName) || '{}';

				return {
					type: 'tool_response',
					name: toolName,
					input: storedInput,
					output:
						typeof outputText === 'string'
							? outputText
							: JSON.stringify(outputText),
					status: data?.['status'] === 'completed' ? 'success' : 'error',
					agent: agentName,
					taskId: currentTaskId?.substring(0, 8) || 'unknown',
				};
			}
		}
	}

	return null;
}

function drawBox(content: string[], width: number = 60): string[] {
	const lines: string[] = [];
	const maxContentWidth = width - 4; // Account for '│ ' and ' │'

	// Top border
	lines.push(`╭─${content[0].padEnd(maxContentWidth, '─')}─╮`);

	// Content lines - handle wrapping and multi-line content
	for (let i = 1; i < content.length; i++) {
		const contentLines = content[i].split('\n');

		for (const contentLine of contentLines) {
			if (contentLine.length <= maxContentWidth) {
				// Line fits, just pad it
				const padded = contentLine.padEnd(maxContentWidth);
				lines.push(`│ ${padded} │`);
			} else {
				// Line is too long, wrap it
				const words = contentLine.split(' ');
				let currentLine = '';

				for (const word of words) {
					// Check if adding this word would exceed the width
					const testLine = currentLine + (currentLine ? ' ' : '') + word;
					if (testLine.length <= maxContentWidth) {
						currentLine = testLine;
					} else {
						// Output current line and start a new one
						if (currentLine) {
							const padded = currentLine.padEnd(maxContentWidth);
							lines.push(`│ ${padded} │`);
						}
						// If the word itself is longer than max width, truncate it
						currentLine =
							word.length > maxContentWidth
								? word.substring(0, maxContentWidth - 3) + '...'
								: word;
					}
				}

				// Output any remaining content
				if (currentLine) {
					const padded = currentLine.padEnd(maxContentWidth);
					lines.push(`│ ${padded} │`);
				}
			}
		}
	}

	// Bottom border
	lines.push(`╰${'─'.repeat(width - 2)}╯`);

	return lines;
}

function formatToolInput(input: string): string {
	try {
		const parsed = JSON.parse(input);
		// Format as a more readable object display
		const entries = Object.entries(parsed);
		if (entries.length === 0) return '{}';
		if (entries.length === 1) {
			const [key, value] = entries[0];
			return `{'${key}': ${JSON.stringify(value)}}`;
		}
		return JSON.stringify(parsed);
	} catch {
		return input;
	}
}

function displayCleanToolEvent(toolEvent: ToolEvent): void {
	// This function now only handles tool responses (Tool messages)
	if (toolEvent.type === 'tool_response') {
		// Get terminal width, default to 80 if not available
		const terminalWidth = process.stdout.columns || 80;
		const boxWidth = Math.max(terminalWidth - 2, 60); // Leave 2 chars margin, minimum 60

		// Format the input for the title
		const formattedInput = formatToolInput(toolEvent.input);
		const toolTitle = `Tool "${toolEvent.name}(${formattedInput})" (${toolEvent.taskId})`;

		// Clean up the output - remove quotes and unescape newlines for better readability
		let cleanOutput = toolEvent.output;
		if (cleanOutput.startsWith("'") && cleanOutput.endsWith("'")) {
			cleanOutput = cleanOutput.slice(1, -1);
		}
		cleanOutput = cleanOutput.replace(/\\n/g, '\n');

		const content = [toolTitle, cleanOutput];

		const boxLines = drawBox(content, boxWidth);
		console.log('');
		boxLines.forEach(line => console.log(colorize('green', line)));
		console.log('');
	}
}

function displayUserMessage(userInput: string): void {
	// Get terminal width, default to 80 if not available
	const terminalWidth = process.stdout.columns || 80;
	const boxWidth = Math.max(terminalWidth - 2, 60); // Leave 2 chars margin, minimum 60

	const content = [`User`, userInput];

	const boxLines = drawBox(content, boxWidth);
	console.log('');
	boxLines.forEach(line => console.log(colorize('blue', line)));
	console.log('');
}

function displayAssistantMessage(message: string): void {
	// Get terminal width, default to 80 if not available
	const terminalWidth = process.stdout.columns || 80;
	const boxWidth = Math.max(terminalWidth - 2, 60); // Leave 2 chars margin, minimum 60

	const content = [`Assistant`, message];

	const boxLines = drawBox(content, boxWidth);
	console.log('');
	boxLines.forEach(line => console.log(colorize('cyan', line)));
	console.log('');
}

function checkForHandoffEvent(
	message: Message,
): {name: string; arguments: string} | null {
	if (!message.parts) return null;

	for (const part of message.parts) {
		if (part.kind === 'data') {
			const data = (part as DataPart).data as Record<string, unknown>;

			// Check for handoff function calls (transfer_to_*)
			if (data?.['name'] && data?.['arguments']) {
				const toolName = data['name'] as string;
				if (toolName.startsWith('transfer_to_')) {
					const args = data['arguments'] as string;
					return {name: toolName, arguments: args};
				}
			}

			// Check for handoff_requested events only (Assistant messages)
			const eventType = data?.['type'] as string;
			if (eventType === 'handoff_requested') {
				const name = data?.['name'] as string;
				const args = data?.['arguments'] as string;
				if (name && args) {
					return {name, arguments: args};
				}
			}
		}
	}

	return null;
}

// --- State ---
let currentTaskId: string | undefined = undefined; // Initialize as undefined
let currentContextId: string | undefined = undefined; // Initialize as undefined
let pendingToolCalls: ToolCall[] = []; // Track pending tool calls awaiting approval
let isWaitingForApproval = false; // Track if we're waiting for user approval
let selectedAgentId: string | undefined = undefined; // Track selected agent ID
let lastDisplayedMessage: string = ''; // Track last displayed message content for delta handling
let currentStreamingLine: string = ''; // Track current line being streamed
let isCurrentlyStreaming: boolean = false; // Track if we're currently in streaming mode
let streamedTaskIds: Set<string> = new Set(); // Track which tasks we've streamed content for
let toolCallInputs: Map<string, string> = new Map(); // Track tool call inputs by tool name
const appConfig = loadAppConfig();
let baseServerUrl = `http://localhost:${appConfig.appPort}`; // Base server URL without agent-specific path
let serverUrl: string; // Will be set after agent selection

// Command line argument parsing
interface CliArgs {
	agentId?: string;
	autoApprove?: boolean;
	userInput?: string;
	baseUrl?: string; // legacy positional
	baseServerUrl?: string; // new named flag
	authToken?: string; // new named flag
}

function parseCliArgs(): CliArgs {
	const args: CliArgs = {};

	for (let i = 2; i < process.argv.length; i++) {
		const arg = process.argv[i];

		if (arg === '--agentId' && i + 1 < process.argv.length) {
			args.agentId = process.argv[++i];
		} else if (arg === '--auto-approve') {
			args.autoApprove = true;
		} else if (arg === '--user-input' && i + 1 < process.argv.length) {
			args.userInput = process.argv[++i];
		} else if (arg === '--base-url' && i + 1 < process.argv.length) {
			args.baseUrl = process.argv[++i];
		} else if (
			(arg === '--baseServerUrl' || arg === '--base-server-url') &&
			i + 1 < process.argv.length
		) {
			args.baseServerUrl = process.argv[++i];
		} else if (
			(arg === '--authToken' || arg === '--auth-token') &&
			i + 1 < process.argv.length
		) {
			args.authToken = process.argv[++i];
		} else if (
			(arg === 'baseServerUrl' || arg === 'base-server-url') &&
			i + 1 < process.argv.length
		) {
			// Support positional style: baseServerUrl <url>
			args.baseServerUrl = process.argv[++i];
		} else if (arg === 'authToken' && i + 1 < process.argv.length) {
			// Support positional style: authToken <token>
			args.authToken = process.argv[++i];
		} else if (!arg.startsWith('--') && !args.baseUrl) {
			// First non-flag argument is base URL for backward compatibility
			args.baseUrl = arg;
		}
	}

	return args;
}

const cliArgs = parseCliArgs();
// Prioritize named flag, then legacy baseUrl, then env override
baseServerUrl = cliArgs.baseServerUrl || cliArgs.baseUrl || baseServerUrl;

// Resolve auth token from CLI or env
const authToken = cliArgs.authToken;

// Inject Authorization header for all requests to baseServerUrl
try {
	const originalFetch = globalThis.fetch.bind(globalThis);
	globalThis.fetch = (input: any, init?: any) => {
		try {
			const url =
				typeof input === 'string' ? input : input?.url ?? String(input);
			if (
				authToken &&
				typeof url === 'string' &&
				baseServerUrl &&
				url.startsWith(baseServerUrl)
			) {
				const headers = new Headers(
					init?.headers ||
						(typeof input?.headers !== 'function'
							? input?.headers
							: undefined) ||
						{},
				);
				if (!headers.has('Authorization')) {
					headers.set('Authorization', `Bearer ${authToken}`);
				}
				return originalFetch(input, {...init, headers});
			}
		} catch {}
		return originalFetch(input, init);
	};
} catch {}

// Context manager import removed - not used

// Debug logging
console.log('🔍 Debug - process.argv:', process.argv);
console.log('🔍 Debug - baseServerUrl:', baseServerUrl);
console.log('🔍 Debug - authToken present:', Boolean(authToken));
console.log('🔍 Debug - cliArgs:', cliArgs);
let client: A2AClient; // Will be initialized asynchronously
let agentName = 'Agent'; // Default, try to get from agent card later

// --- Agent Loading and Selection ---
interface AgentInfo {
	id: string;
	name: string;
	description: string;
	systemPrompt: string;
}

async function loadAvailableAgents(): Promise<AgentInfo[]> {
	try {
		// Fetch agents from the /agents endpoint against configured baseServerUrl
		const response = await fetch(
			`${baseServerUrl}/agents`,
			authToken
				? {headers: {Authorization: `Bearer ${authToken}`}}
				: (undefined as any),
		);
		if (response.ok) {
			const agents = await response.json();
			// console.log(`📋 Loaded ${agents.length} agents from /agents endpoint`);

			return agents.map((agent: any) => ({
				id: agent.id,
				name: agent.name,
				description:
					agent.handoffDescription || agent.instructions || agent.name,
				systemPrompt: agent.instructions,
			}));
		} else {
			throw new Error(
				`Failed to fetch agents from endpoint (${response.status})`,
			);
		}
	} catch (error) {
		console.error(`Error fetching agents from endpoint: ${error}`);
		throw new Error(
			`Unable to load agents: ${
				error instanceof Error ? error.message : 'Unknown error'
			}`,
		);
	}
}

async function selectAgent(): Promise<string> {
	const agents = await loadAvailableAgents();

	// If agentId provided via CLI, validate and use it
	if (cliArgs.agentId) {
		const agent = agents.find(a => a.id === cliArgs.agentId);
		if (!agent) {
			console.error(
				colorize(
					'red',
					`❌ Agent ID ${cliArgs.agentId} not found in agents.jsonl`,
				),
			);
			console.log(colorize('dim', 'Available agent IDs:'));
			agents.forEach(a => console.log(`  ${a.id} - ${a.name}`));
			process.exit(1);
		}
		console.log(
			colorize('green', `✓ Using CLI specified agent: ${agent.name}`),
		);
		console.log(colorize('dim', `   ${agent.description}`));
		return agent.id;
	}

	if (agents.length === 1) {
		console.log(
			colorize('green', `✓ Using single available agent: ${agents[0].name}`),
		);
		return agents[0].id;
	}

	console.log(colorize('bright', '\n🤖 Available Agents:'));
	console.log(colorize('dim', 'Please select an agent to chat with:\n'));

	agents.forEach((agent, index) => {
		console.log(
			`${colorize('cyan', `${index + 1}.`)} ${colorize(
				'bright',
				agent.name,
			)} ${colorize('gray', `(${agent.id})`)}`,
		);
		console.log(`   ${colorize('dim', agent.description)}`);
		if (index < agents.length - 1) console.log(); // Add spacing between agents
	});

	console.log();

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
					console.log(colorize('green', `✓ Selected: ${selectedAgent.name}`));
					console.log(colorize('dim', `   ${selectedAgent.description}\n`));

					resolve(selectedAgent.id);
				},
			);
		};

		askForSelection();
	});
}

// --- Readline Setup ---
const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
	prompt: colorize('cyan', 'You: '),
});

// --- Response Handling ---
// Function now accepts the unwrapped event payload directly
async function printAgentEvent(
	event: TaskStatusUpdateEvent | TaskArtifactUpdateEvent,
) {
	const timestamp = new Date().toLocaleTimeString();
	const prefix = colorize('magenta', `\n${agentName} [${timestamp}]:`);

	// Check if it's a TaskStatusUpdateEvent
	if (event.kind === 'status-update') {
		const update = event as TaskStatusUpdateEvent; // Cast for type safety
		const state = update.status.state;
		let stateEmoji = '❓';
		let stateColor: keyof typeof colors = 'yellow';

		switch (state) {
			case 'working':
				stateEmoji = '⏳';
				stateColor = 'blue';
				break;
			case 'input-required':
				stateEmoji = '🤔';
				stateColor = 'yellow';
				break;
			case 'completed':
				stateEmoji = '✅';
				stateColor = 'green';
				break;
			case 'canceled':
				stateEmoji = '⏹️';
				stateColor = 'gray';
				break;
			case 'failed':
				stateEmoji = '❌';
				stateColor = 'red';
				break;
			default:
				stateEmoji = 'ℹ️'; // For other states like submitted, rejected etc.
				stateColor = 'dim';
				break;
		}

		// Always show status line for all updates (streaming or not)

		// Detect and surface handoff events even during streaming
		let handoffEvent: string | undefined;
		if (update.status.message?.parts) {
			for (const p of update.status.message.parts) {
				if (p.kind === 'data') {
					const d = (p as DataPart).data as Record<string, unknown> | undefined;
					const t = (d?.['type'] as string) || undefined;
					if (t === 'handoff_requested' || t === 'handoff_occurred') {
						handoffEvent = t;
						break;
					}
				}
			}
		}
		// Check if this is a raw_model_stream_event (suppress completely)
		let isRawModelStreamEvent = false;
		if (update.status.message?.parts) {
			for (const p of update.status.message.parts) {
				if (p.kind === 'data') {
					const d = (p as DataPart).data as Record<string, unknown> | undefined;
					const t = (d?.['type'] as string) || undefined;
					if (t === 'model') {
						isRawModelStreamEvent = true;
						break;
					}
				}
			}
		}

		// Suppress raw model stream events and most status updates
		// Only show status for important state changes, but not for tool approval states
		const isToolApprovalState =
			state === 'input-required' &&
			update.status.message?.parts?.some(
				p =>
					p.kind === 'data' &&
					(p as DataPart).data &&
					typeof (p as DataPart).data === 'object' &&
					((p as DataPart).data as Record<string, unknown>)?.['name'], // This indicates a tool call
			);

		if (
			!isRawModelStreamEvent &&
			!isToolApprovalState &&
			(state === 'completed' || state === 'failed' || handoffEvent)
		) {
			const statusLine =
				`${prefix} ${stateEmoji} Status: ${colorize(
					stateColor,
					state,
				)} (Task: ${update.taskId}, Context: ${update.contextId}) ${
					update.final ? colorize('bright', '[FINAL]') : ''
				}` +
				(handoffEvent
					? ` ${colorize('blue', '[handoff]')} ${handoffEvent}`
					: '');
			console.log(statusLine);
		}

		// Clear task ID when task is final and completed (not just working or input-required)
		if (update.final && state === 'completed') {
			// If we were streaming, close the box
			if (isCurrentlyStreaming) {
				// Close the streaming box
				const terminalWidth = process.stdout.columns || 80;
				const boxWidth = Math.max(terminalWidth - 2, 60);
				const currentLineLength = currentStreamingLine.length;
				const padding = ' '.repeat(
					Math.max(0, boxWidth - 4 - currentLineLength),
				);
				process.stdout.write(
					colorize('cyan', `${padding} │\n╰${'─'.repeat(boxWidth - 2)}╯\n`),
				);
				currentStreamingLine = ''; // Reset for next streaming session
			}

			currentTaskId = undefined;
			isWaitingForApproval = false;
			pendingToolCalls = [];
			lastDisplayedMessage = ''; // Reset for next task
			isCurrentlyStreaming = false; // Reset streaming state
			// Note: Don't clean streamedTaskIds here - let the task event handle cleanup
		}

		// Update currentTaskId and currentContextId from status update
		// Don't update task ID if we just cleared it for a final completed task
		if (
			update.taskId &&
			update.taskId !== currentTaskId &&
			!(update.final && state === 'completed')
		) {
			currentTaskId = update.taskId;
			lastDisplayedMessage = ''; // Reset for new task
			isCurrentlyStreaming = false; // Reset streaming state for new task
		}
		if (update.contextId && update.contextId !== currentContextId) {
			console.log(
				colorize(
					'dim',
					`   Context ID updated from ${currentContextId || 'N/A'} to ${
						update.contextId
					}`,
				),
			);
			currentContextId = update.contextId;
		}

		// Skip message processing for raw_model_stream_event
		if (update.status.message && !isRawModelStreamEvent) {
			// Check if this is a tool call approval request (data-based only)
			if (state === 'input-required' && update.status.message.parts) {
				for (const part of update.status.message.parts) {
					if (part.kind === 'data') {
						const dataPart = part as DataPart;
						// Silently handle tool approval without displaying verbose output
						await handleToolCallFromStatusData(
							dataPart.data as unknown,
							update.taskId,
							update.contextId,
						);
					}
				}
				// Return early to prevent further processing of approval messages
				return;
			}

			// Check if this is a tool call or tool output
			const toolEvent = checkForToolEvent(update.status.message);
			if (toolEvent) {
				if (toolEvent.type === 'tool_call') {
					// Tool calls are Assistant messages
					displayAssistantMessage(
						`${toolEvent.name}("${toolEvent.arguments}")`,
					);
				} else {
					// Tool responses are Tool messages
					displayCleanToolEvent(toolEvent);
				}
			} else {
				// Check if this is a handoff event (should be displayed as Assistant message)
				const handoffEvent = checkForHandoffEvent(update.status.message);
				if (handoffEvent) {
					displayAssistantMessage(
						`${handoffEvent.name}("${handoffEvent.arguments}")`,
					);
				} else {
					// Handle streaming for output_text_delta
					const hasOutputTextDelta = update.status.message.parts?.some(
						p =>
							p.kind === 'data' &&
							(p as DataPart).data &&
							typeof (p as DataPart).data === 'object' &&
							((p as DataPart).data as Record<string, unknown>)?.['type'] ===
								'output_text_delta',
					);

					if (hasOutputTextDelta) {
						// For streaming, display content with proper box borders
						if (!isCurrentlyStreaming) {
							isCurrentlyStreaming = true;
							streamedTaskIds.add(update.taskId);
							currentStreamingLine = ''; // Reset current line tracking
							// Start the streaming box
							const terminalWidth = process.stdout.columns || 80;
							const boxWidth = Math.max(terminalWidth - 2, 60);
							console.log('');
							console.log(
								colorize('cyan', `╭─Assistant${'─'.repeat(boxWidth - 12)}╮`),
							);
							process.stdout.write(colorize('cyan', '│ '));
						}
						// Display streamed content with borders
						printMessageContentStreaming(update.status.message, true);
					} else if (
						state === 'working' &&
						isCurrentlyStreaming &&
						!hasOutputTextDelta
					) {
						// End of streaming detected - display the collected content
						if (lastDisplayedMessage && lastDisplayedMessage.trim()) {
							displayAssistantMessage(lastDisplayedMessage.trim());
							lastDisplayedMessage = '';
							isCurrentlyStreaming = false;
						}
					} else if (state === 'completed' && isCurrentlyStreaming) {
						// End streaming - display the collected content in a proper box
						if (lastDisplayedMessage && lastDisplayedMessage.trim()) {
							displayAssistantMessage(lastDisplayedMessage.trim());
						}
						isCurrentlyStreaming = false;
					} else {
						// Regular non-streaming message - check if it has text content
						const textContent = update.status.message.parts?.find(
							p => p.kind === 'text',
						)?.text;
						if (textContent && textContent.trim()) {
							displayAssistantMessage(textContent.trim());
						} else {
							// Fallback for other types of content
							printMessageContentStreaming(update.status.message, false);
						}
					}
				}
			}
		}
	}
	// Check if it's a TaskArtifactUpdateEvent
	else if (event.kind === 'artifact-update') {
		const update = event as TaskArtifactUpdateEvent; // Cast for type safety

		// Check if this is a tool call artifact
		const isToolCallArtifact = update.artifact.parts.some(
			part =>
				part.kind === 'data' &&
				(part as DataPart).data &&
				typeof (part as DataPart).data === 'object' &&
				(part as DataPart).data['toolCall'],
		);

		if (isToolCallArtifact) {
			// Only process tool call artifacts if we have an active task
			if (currentTaskId) {
				// Show artifact received message first
				console.log(
					`${prefix} 📄 Artifact Received: ${
						update.artifact.name || '(unnamed)'
					} (ID: ${update.artifact.artifactId}, Task: ${
						update.taskId
					}, Context: ${update.contextId})`,
				);
				// Use special formatting for tool call artifacts
				await printToolCallArtifact(update.artifact);
			}
			// Silently ignore tool call artifacts for completed tasks
		} else {
			// Use standard artifact formatting for other artifacts
			console.log(
				`${prefix} 📄 Artifact Received: ${
					update.artifact.name || '(unnamed)'
				} (ID: ${update.artifact.artifactId}, Task: ${
					update.taskId
				}, Context: ${update.contextId})`,
			);
			// Create a temporary message-like structure to reuse printMessageContent
			printMessageContent({
				messageId: generateId(), // Dummy messageId
				kind: 'message', // Dummy kind
				role: 'agent', // Assuming artifact parts are from agent
				parts: update.artifact.parts,
				taskId: update.taskId,
				contextId: update.contextId,
			});
		}
	} else {
		// This case should ideally not be reached if called correctly
		console.log(
			prefix,
			colorize('yellow', 'Received unknown event type in printAgentEvent:'),
			event,
		);
	}
}

function printMessageContent(message: Message) {
	message.parts.forEach((part: Part, index: number) => {
		// Added explicit Part type
		if (part.kind === 'text') {
			// Check kind property - display in Assistant box
			displayAssistantMessage(part.text);
		} else if (part.kind === 'file') {
			// Check kind property
			const filePart = part as FilePart;
			console.log(
				`${colorize('red', `  Part ${index + 1}:`)} ${colorize(
					'blue',
					'📄 File:',
				)} Name: ${filePart.file.name || 'N/A'}, Type: ${
					filePart.file.mimeType || 'N/A'
				}, Source: ${
					'bytes' in filePart.file ? 'Inline (bytes)' : filePart.file.uri
				}`,
			);
		} else if (part.kind === 'data') {
			const dataPart = part as DataPart;
			const data = dataPart.data as Record<string, unknown> | undefined;
			const dataType = (data?.['type'] as string) || undefined;
			const functionName = (data?.['name'] as string) || undefined;

			// Suppress provider model events
			if (dataType === 'model') {
				return;
			}

			// Suppress handoff function call results (transfer_to_*)
			if (
				dataType === 'function_call_result' &&
				functionName?.startsWith('transfer_to_')
			) {
				return;
			}

			// For non-streaming contexts, show other data parts verbosely, except handoff signals
			console.log(
				`${colorize('red', `  Part ${index + 1}:`)} ${colorize(
					'yellow',
					'📊 Data:',
				)}`,
				JSON.stringify(dataPart.data, null, 2),
			);
		} else {
			console.log(
				`${colorize('red', `  Part ${index + 1}:`)} ${colorize(
					'yellow',
					'Unsupported part kind:',
				)}`,
				part,
			);
		}
	});
}

function printMessageContentStreaming(
	message: Message,
	isStreaming: boolean = false,
) {
	message.parts.forEach((part: Part, index: number) => {
		if (part.kind === 'text') {
			if (isStreaming) {
				// For streaming, silently collect the content without displaying
				const currentText = part.text;
				if (currentText !== lastDisplayedMessage) {
					lastDisplayedMessage = currentText;
				}
			} else {
				// For non-streaming, show complete message in Assistant box
				displayAssistantMessage(part.text);
				lastDisplayedMessage = part.text;
			}
		} else if (part.kind === 'file') {
			const filePart = part as FilePart;
			console.log(
				`${colorize('red', `  Part ${index + 1}:`)} ${colorize(
					'blue',
					'📄 File:',
				)} Name: ${filePart.file.name || 'N/A'}, Type: ${
					filePart.file.mimeType || 'N/A'
				}, Source: ${
					'bytes' in filePart.file ? 'Inline (bytes)' : filePart.file.uri
				}`,
			);
		} else if (part.kind === 'data') {
			const dataPart = part as DataPart;
			const data = dataPart.data as Record<string, unknown> | undefined;
			const dataType = (data?.['type'] as string) || undefined;
			const functionName = (data?.['name'] as string) || undefined;

			if (dataType === 'model') {
				// Suppress provider model events
				return;
			}

			// Suppress handoff function call results (transfer_to_*)
			if (
				dataType === 'function_call_result' &&
				functionName?.startsWith('transfer_to_')
			) {
				return;
			}

			if (dataType === 'output_text_delta') {
				const delta = (data?.['delta'] as string) || '';
				if (delta) {
					const terminalWidth = process.stdout.columns || 80;
					const boxWidth = Math.max(terminalWidth - 2, 60);

					// Handle line breaks properly with box borders
					if (delta.includes('\n')) {
						const lines = delta.split('\n');

						for (let i = 0; i < lines.length; i++) {
							if (i > 0) {
								// End current line with proper padding and start new line with border
								const currentLineLength =
									currentStreamingLine.length + (i === 0 ? lines[0].length : 0);
								const padding = ' '.repeat(
									Math.max(0, boxWidth - 4 - currentLineLength),
								);
								process.stdout.write(colorize('cyan', `${padding} │\n│ `));
								currentStreamingLine = ''; // Reset for new line
							}
							if (lines[i]) {
								process.stdout.write(lines[i]);
								currentStreamingLine += lines[i];
							}
						}
					} else {
						process.stdout.write(delta);
						currentStreamingLine += delta;
					}
					lastDisplayedMessage = `${lastDisplayedMessage}${delta}`;
				}
				return;
			}
			// Fallback: show data
			console.log(
				`${colorize('red', `  Part ${index + 1}:`)} ${colorize(
					'yellow',
					'📊 Data:',
				)}`,
				JSON.stringify(dataPart.data, null, 2),
			);
		} else {
			console.log(
				`${colorize('red', `  Part ${index + 1}:`)} ${colorize(
					'yellow',
					'Unsupported part kind:',
				)}`,
				part,
			);
		}
	});
}

async function printToolCallArtifact(artifact: any) {
	// Suppress verbose approval display - tool boxes will show the information cleanly

	let toolName = '';
	let toolCallData: any = null;

	// Extract tool call information first
	artifact.parts.forEach((part: Part, _index: number) => {
		if (part.kind === 'data') {
			const dataPart = part as DataPart;
			const data = dataPart.data as any;

			if (data.toolCall) {
				toolName = data.toolCall.name;
				toolCallData = {
					id: data.toolCall.id,
					name: data.toolCall.name,
					parameters: data.toolCall.parameters,
					artifactId: artifact.artifactId,
				};
			}
		}
	});

	// Check for auto-preferences BEFORE setting approval state
	if (toolName && toolCallData) {
		const autoPreference = checkAutoPreference(toolName);
		if (autoPreference || cliArgs.autoApprove) {
			const reason = cliArgs.autoApprove
				? 'CLI auto-approve flag'
				: `Auto-${autoPreference} based on saved preference`;
			const decision = (cliArgs.autoApprove ? 'approve' : autoPreference!) as
				| 'approve'
				| 'reject';

			// Check if we have an active task before proceeding with auto-approval
			if (!currentTaskId) {
				// Silently ignore tool call artifacts for completed tasks
				return;
			}

			// Set approval state and tool call data
			pendingToolCalls = [toolCallData];
			isWaitingForApproval = true;

			// Automatically handle the tool call - suppress verbose output
			await handleToolApproval(decision, reason, toolCallData);
			return;
		}
	}

	// Clear any existing pending tool calls and set approval state
	pendingToolCalls = [];
	isWaitingForApproval = true;

	artifact.parts.forEach((part: Part, _index: number) => {
		if (part.kind === 'data') {
			const dataPart = part as DataPart;
			const data = dataPart.data as any;

			if (data.toolCall) {
				console.log(
					colorize(
						'yellow',
						`\n  🛠️  Tool: ${colorize('bright', data.toolCall.name)}`,
					),
				);
				console.log(colorize('dim', `     ID: ${data.toolCall.id}`));

				if (
					data.toolCall.parameters &&
					Object.keys(data.toolCall.parameters).length > 0
				) {
					console.log(colorize('dim', `     Parameters:`));
					Object.entries(data.toolCall.parameters).forEach(([key, value]) => {
						console.log(
							colorize('dim', `       ${key}: ${JSON.stringify(value)}`),
						);
					});
				}

				// Store the tool call for approval handling
				pendingToolCalls.push({
					id: data.toolCall.id,
					name: data.toolCall.name,
					parameters: data.toolCall.parameters,
					artifactId: artifact.artifactId,
				});
			}

			if (data.authRequired) {
				console.log(colorize('red', `\n  🔐 Approval Required:`));
				console.log(colorize('dim', `     Type: ${data.authRequired.type}`));
				if (data.authRequired.agent) {
					console.log(
						colorize('dim', `     Agent: ${data.authRequired.agent}`),
					);
				}
				if (data.authRequired.reason) {
					console.log(
						colorize('dim', `     Reason: ${data.authRequired.reason}`),
					);
				}
			}
		}
	});

	// Tool options are handled automatically for auto-approve scenarios
}

// --- Tool Call Parsing from Status Data (preferred) ---
type RawToolCallPayload = {
	id?: string;
	name?: string;
	arguments?: unknown;
	args?: unknown;
	parameters?: Record<string, unknown>;
	artifactId?: string;
	tool?: {name?: string} | undefined;
};

async function handleToolCallFromStatusData(
	data: unknown,
	_taskId: string,
	_contextId: string,
): Promise<void> {
	try {
		// Expecting shape similar to item.rawItem { id?, name, arguments }
		const payload = (data ?? {}) as RawToolCallPayload;
		const toolName: string | undefined = payload.name || payload.tool?.name;
		const rawArgs: unknown =
			payload.arguments ?? payload.args ?? payload.parameters;

		if (!toolName) {
			return; // Not a tool-call status payload; ignore
		}

		// Normalize arguments to object
		let parameters: Record<string, unknown> = {};
		if (rawArgs !== null && rawArgs !== undefined) {
			if (typeof rawArgs === 'string') {
				try {
					parameters = JSON.parse(rawArgs);
				} catch {
					// Fallback: wrap raw string
					parameters = {value: rawArgs};
				}
			} else if (typeof rawArgs === 'object') {
				parameters = rawArgs as Record<string, unknown>;
			}
		}

		// Compose a synthetic artifactId if not present
		const artifactId = payload.artifactId || `tool-call-${Date.now()}`;
		const callId = payload.id || `call_${Date.now()}`;

		// Suppress verbose approval display - tool boxes will show the information cleanly

		const toolCall: ToolCall = {
			id: callId,
			name: toolName,
			parameters,
			artifactId,
		};

		pendingToolCalls.push(toolCall);
		isWaitingForApproval = true;

		// Auto preferences
		const autoPreference = checkAutoPreference(toolName);
		if (autoPreference || cliArgs.autoApprove) {
			const reason = cliArgs.autoApprove
				? 'CLI auto-approve flag'
				: `Auto-${autoPreference} based on saved preference`;
			const decision = (cliArgs.autoApprove ? 'approve' : autoPreference!) as
				| 'approve'
				| 'reject';
			// Suppress verbose auto-approval messages - tool boxes will show the clean information
			await handleToolApproval(decision, reason, toolCall);
			return;
		}

		// Tool options are handled automatically for auto-approve scenarios
	} catch (_err) {
		console.log(colorize('red', 'Failed to process tool-call status data'));
	}
}

// --- Tool Call Preference Management ---
function loadToolPreferences(): Record<
	string,
	{autoApprove: boolean; autoReject: boolean}
> {
	const timestepPaths = getTimestepPaths();
	const preferencesFile = path.join(
		timestepPaths.configDir,
		'preferences.json',
	);
	try {
		if (fs.existsSync(preferencesFile)) {
			const preferencesData = JSON.parse(
				fs.readFileSync(preferencesFile, 'utf8'),
			);
			return preferencesData.toolPreferences || {};
		}
	} catch (error) {
		console.error('🔍 Error loading tool preferences:', error);
	}
	return {};
}

function saveToolPreferences(
	preferences: Record<string, {autoApprove: boolean; autoReject: boolean}>,
): boolean {
	const timestepPaths = getTimestepPaths();
	const preferencesFile = path.join(
		timestepPaths.configDir,
		'preferences.json',
	);
	try {
		let preferencesData: any = {};
		if (fs.existsSync(preferencesFile)) {
			preferencesData = JSON.parse(fs.readFileSync(preferencesFile, 'utf8'));
		}

		preferencesData.toolPreferences = preferences;

		// Ensure directory exists
		const dir = path.dirname(preferencesFile);
		if (!fs.existsSync(dir)) {
			fs.mkdirSync(dir, {recursive: true});
		}

		fs.writeFileSync(preferencesFile, JSON.stringify(preferencesData, null, 2));
		return true;
	} catch (error) {
		console.error('🔍 Error saving tool preferences:', error);
		console.log(
			colorize(
				'red',
				'⚠️  Failed to save tool preferences. They will not persist across sessions.',
			),
		);
		return false;
	}
}

function getToolPreference(toolName: string): {
	autoApprove: boolean;
	autoReject: boolean;
} {
	const preferences = loadToolPreferences();
	return preferences[toolName] || {autoApprove: false, autoReject: false};
}

function setToolPreference(
	toolName: string,
	autoApprove: boolean,
	autoReject: boolean,
): boolean {
	const preferences = loadToolPreferences();
	preferences[toolName] = {autoApprove, autoReject};
	const success = saveToolPreferences(preferences);

	if (success) {
		const action = autoApprove ? 'auto-approve' : 'auto-reject';
		console.log(colorize('green', `✅ Set ${action} for tool: ${toolName}`));
	}

	return success;
}

function checkAutoPreference(toolName: string): 'approve' | 'reject' | null {
	const preference = getToolPreference(toolName);
	if (preference.autoApprove) return 'approve';
	if (preference.autoReject) return 'reject';
	return null;
}

async function executeToolCall(toolCall: ToolCall): Promise<string> {
	// No longer execute tools directly - just return a placeholder
	// The actual execution will be handled by the agent executor
	console.log(
		colorize(
			'dim',
			`   🔧 Tool call approved: ${
				toolCall.name
			} with parameters: ${JSON.stringify(toolCall.parameters)}`,
		),
	);
	return `Tool ${toolCall.name} approved for execution`;
}

async function handleToolApproval(
	decision: string,
	reason?: string,
	specificToolCall?: ToolCall,
): Promise<void> {
	// Determine which tool call to process
	const toolCallToProcess = specificToolCall || pendingToolCalls[0];

	if (!toolCallToProcess) {
		console.log(colorize('red', 'No pending tool calls to approve.'));
		return;
	}

	// Check if we have an active task before proceeding
	if (!currentTaskId) {
		console.log(
			colorize(
				'yellow',
				'⚠️ No active task found. Tool call cannot be processed.',
			),
		);
		// Clear pending tool calls and reset approval state
		pendingToolCalls = [];
		isWaitingForApproval = false;
		return;
	}

	const isApproved = decision.toLowerCase().startsWith('approve');
	const isAutoApproval =
		reason?.includes('auto-approve') || reason?.includes('CLI auto-approve');

	// Only show verbose output if not auto-approving
	if (!isAutoApproval) {
		console.log(
			colorize('cyan', `\n🔧 Processing tool call decision: ${decision}`),
		);
	}

	let toolResult: string;

	if (isApproved) {
		if (!isAutoApproval) {
			console.log(
				colorize('green', `✅ Approving tool call: ${toolCallToProcess.name}`),
			);
			if (reason) {
				console.log(colorize('dim', `   Reason: ${reason}`));
			}
		}

		// Execute the tool call
		toolResult = await executeToolCall(toolCallToProcess);

		if (!isAutoApproval) {
			console.log(colorize('green', `   Tool result: ${toolResult}`));
		}
	} else {
		console.log(
			colorize('red', `❌ Rejecting tool call: ${toolCallToProcess.name}`),
		);
		if (reason) {
			console.log(colorize('dim', `   Reason: ${reason}`));
		}
		toolResult = `Tool call rejected by user${reason ? `: ${reason}` : ''}`;
	}

	// Create tool response message with structured data
	// The agent executor will detect the toolCallResponse data and add it as a function_call_result
	const toolResponseMessage: Message = {
		messageId: generateId(),
		kind: 'message',
		role: 'user',
		parts: [
			{
				kind: 'data',
				data: {
					toolCallResponse: {
						callId: toolCallToProcess.id,
						artifactId: toolCallToProcess.artifactId, // Reference to original tool call artifact
						status: decision === 'approve' ? 'approved' : 'rejected',
						decision: decision,
						reason: reason,
						result: toolResult,
						executedAt: new Date().toISOString(),
					},
				},
			},
		],
		taskId: currentTaskId,
		contextId: currentContextId,
	};

	// Send the tool response back to continue the conversation
	const params: MessageSendParams = {
		message: toolResponseMessage,
	};

	try {
		// Suppress verbose messaging for auto-approved tools
		if (!isAutoApproval) {
			console.log(
				colorize(
					'cyan',
					`\n📤 Sending tool response to continue conversation...`,
				),
			);
		}
		const stream = client.sendMessageStream(params);

		// Process the response stream
		for await (const event of stream) {
			if (event.kind === 'status-update' || event.kind === 'artifact-update') {
				const typedEvent = event as
					| TaskStatusUpdateEvent
					| TaskArtifactUpdateEvent;
				await printAgentEvent(typedEvent);
			} else if (event.kind === 'task') {
				const task = event as Task;

				// For completed tasks, close the streaming box first, then log completion
				if (task.status.state === 'completed') {
					if (isCurrentlyStreaming) {
						const terminalWidth = process.stdout.columns || 80;
						const boxWidth = Math.max(terminalWidth - 2, 60);
						const currentLineLength = currentStreamingLine.length;
						const padding = ' '.repeat(
							Math.max(0, boxWidth - 4 - currentLineLength),
						);
						process.stdout.write(
							colorize('cyan', `${padding} │\n╰${'─'.repeat(boxWidth - 2)}╯\n`),
						);
						currentStreamingLine = '';
						isCurrentlyStreaming = false;
					}
					console.log(
						colorize('blue', `\n📋 Task Update: ${task.status.state}`),
					);
				} else {
					console.log(
						colorize('blue', `\n📋 Task Update: ${task.status.state}`),
					);
				}

				// For non-completed tasks, display any accumulated streaming content
				if (task.status.message) {
					if (task.status.state === 'completed') {
						// Skip showing the message content for completed tasks - it was streamed already
					} else {
						// Display the task message in an Assistant box
						const messageText =
							task.status.message.parts
								?.filter(part => part.kind === 'text')
								?.map(part => part.text)
								?.join(' ') || '';
						if (messageText) {
							displayAssistantMessage(messageText);
						}
					}
				}
			}
		}

		// Remove the processed tool call but keep any newer ones that may have been added
		pendingToolCalls = pendingToolCalls.filter(
			tc => tc.id !== toolCallToProcess.id,
		);

		// Only reset approval state if no more tool calls are pending
		if (pendingToolCalls.length === 0) {
			isWaitingForApproval = false;
		} else {
			// Keep waiting for approval for remaining tool calls
		}
	} catch (error: any) {
		console.error(
			colorize('red', 'Error sending tool response:'),
			error.message,
		);
		// Clear pending tool calls and reset approval state on error
		pendingToolCalls = [];
		isWaitingForApproval = false;
	}
}

// --- Agent Card Fetching ---
async function fetchAndDisplayAgentCard() {
	// Use the client's getAgentCard method.
	// The client was initialized with serverUrl, which is the agent's base URL.
	console.log(
		colorize(
			'dim',
			`Attempting to fetch agent card from agent at: ${serverUrl}`,
		),
	);
	try {
		// client.getAgentCard() uses the agentBaseUrl provided during client construction
		// getAgentCard takes no options in current SDK; rely on client default headers
		const card: AgentCard = await client.getAgentCard();
		agentName = card.name || 'Agent'; // Update global agent name
		console.log(colorize('green', `✓ Agent Card Found:`));
		console.log(`  Name:        ${colorize('bright', agentName)}`);
		if (card.description) {
			console.log(`  Description: ${card.description}`);
		}
		console.log(`  Version:     ${card.version || 'N/A'}`);
		if (card.capabilities?.streaming) {
			console.log(`  Streaming:   ${colorize('green', 'Supported')}`);
		} else {
			console.log(
				`  Streaming:   ${colorize(
					'yellow',
					'Not Supported (or not specified)',
				)}`,
			);
		}
		// Update prompt prefix to use the fetched name
		// The prompt is set dynamically before each rl.prompt() call in the main loop
		// to reflect the current agentName if it changes (though unlikely after initial fetch).
	} catch (error: any) {
		console.log(colorize('yellow', `⚠️ Error fetching or parsing agent card`));
		throw error;
	}
}

// --- Input Processing ---
async function processInput(
	input: string,
	isInteractive: boolean = true,
): Promise<void> {
	if (!input) {
		if (isInteractive) rl.prompt();
		return;
	}

	if (input.toLowerCase() === '/new') {
		currentTaskId = undefined;
		currentContextId = undefined; // Reset contextId on /new
		pendingToolCalls = []; // Clear pending tool calls
		isWaitingForApproval = false; // Reset approval state
		console.log(
			colorize(
				'bright',
				`✨ Starting new session with ${agentName}. Task and Context IDs are cleared.`,
			),
		);
		if (isInteractive) rl.prompt();
		return;
	}

	if (input.toLowerCase() === '/exit') {
		if (isInteractive) rl.close();
		return;
	}

	// Handle tool call approval/rejection
	if (isWaitingForApproval && pendingToolCalls.length > 0) {
		const lowerInput = input.toLowerCase();

		if (lowerInput.startsWith('approve') || lowerInput.startsWith('reject')) {
			const parts = input.split(' - ');
			const decision = parts[0];
			const reason = parts.length > 1 ? parts[1] : undefined;

			await handleToolApproval(decision, reason);
			if (isInteractive) rl.prompt();
			return;
		} else if (lowerInput === 'auto-approve') {
			const toolCall = pendingToolCalls[0];
			setToolPreference(toolCall.name, true, false);
			await handleToolApproval('approve', 'Auto-approve enabled for this tool');
			if (isInteractive) rl.prompt();
			return;
		} else if (lowerInput === 'auto-reject') {
			const toolCall = pendingToolCalls[0];
			setToolPreference(toolCall.name, false, true);
			await handleToolApproval('reject', 'Auto-reject enabled for this tool');
			if (isInteractive) rl.prompt();
			return;
		} else if (lowerInput === 'show-params') {
			const toolCall = pendingToolCalls[0];
			console.log(
				colorize('cyan', `\n📋 Current Parameters for ${toolCall.name}:`),
			);
			Object.entries(toolCall.parameters).forEach(([key, value]) => {
				console.log(colorize('dim', `  ${key}: ${JSON.stringify(value)}`));
			});
			if (isInteractive) rl.prompt();
			return;
		} else if (lowerInput.startsWith('modify ')) {
			const modifyCommand = input.substring(7); // Remove "modify "
			const equalIndex = modifyCommand.indexOf('=');
			if (equalIndex === -1) {
				console.log(
					colorize(
						'red',
						'Invalid modify command. Use: modify <param>=<value>',
					),
				);
				if (isInteractive) rl.prompt();
				return;
			}

			const paramName = modifyCommand.substring(0, equalIndex).trim();
			const paramValue = modifyCommand.substring(equalIndex + 1).trim();

			// Input validation
			if (!paramName || !paramValue) {
				console.log(
					colorize('red', 'Parameter name and value cannot be empty.'),
				);
				if (isInteractive) rl.prompt();
				return;
			}

			// Basic sanitization - remove potentially dangerous characters
			const sanitizedParamName = paramName.replace(/[^a-zA-Z0-9_]/g, '');
			if (sanitizedParamName !== paramName) {
				console.log(
					colorize(
						'red',
						'Parameter name contains invalid characters. Only letters, numbers, and underscores are allowed.',
					),
				);
				if (isInteractive) rl.prompt();
				return;
			}

			const toolCall = pendingToolCalls[0];
			toolCall.parameters[paramName] = paramValue;

			console.log(
				colorize(
					'green',
					`✅ Modified parameter: ${paramName} = ${paramValue}`,
				),
			);
			console.log(
				colorize('cyan', `\n📋 Updated Parameters for ${toolCall.name}:`),
			);
			Object.entries(toolCall.parameters).forEach(([key, value]) => {
				console.log(colorize('dim', `  ${key}: ${JSON.stringify(value)}`));
			});
			if (isInteractive) rl.prompt();
			return;
		} else {
			console.log(
				colorize('yellow', 'Please respond with one of the available options:'),
			);
			console.log(colorize('dim', '  • approve [reason]'));
			console.log(colorize('dim', '  • reject [reason]'));
			console.log(colorize('dim', '  • modify <param>=<value>'));
			console.log(colorize('dim', '  • auto-approve'));
			console.log(colorize('dim', '  • auto-reject'));
			console.log(colorize('dim', '  • show-params'));
			if (isInteractive) rl.prompt();
			return;
		}
	}

	// Construct params for sendMessageStream
	const messageId = generateId(); // Generate a unique message ID

	const messagePayload: Message = {
		messageId: messageId,
		kind: 'message', // Required by Message interface
		role: 'user',
		parts: [
			{
				kind: 'text', // Required by TextPart interface
				text: input,
			},
		],
	};

	// For task-generating agents, don't send taskId for new messages
	// Each message should create a new task, not modify the existing completed task
	// Only send contextId to maintain conversation history
	// Conditionally add contextId to the message payload
	if (currentContextId) {
		messagePayload.contextId = currentContextId;
	}

	const params: MessageSendParams = {
		message: messagePayload,
		headers: authToken ? {Authorization: `Bearer ${authToken}`} : undefined,
	} as any;

	try {
		// Display the user message clearly
		displayUserMessage(input);

		// Use sendMessageStream
		const stream = client.sendMessageStream(params);

		// Iterate over the events from the stream
		for await (const event of stream) {
			const timestamp = new Date().toLocaleTimeString(); // Get fresh timestamp for each event
			const prefix = colorize('magenta', `\n${agentName} [${timestamp}]:`);

			// Process artifact-update events

			if (event.kind === 'status-update' || event.kind === 'artifact-update') {
				const typedEvent = event as
					| TaskStatusUpdateEvent
					| TaskArtifactUpdateEvent;
				await printAgentEvent(typedEvent);

				// Task clearing logic moved to the status display section in printAgentEvent
			} else if (event.kind === 'message') {
				// Task-generating agents should never send message events - only Task objects
				throw new Error(
					`❌ ERROR: Received unexpected 'message' event. Task-generating agents should only send Task objects, not Message events. Event details: ${JSON.stringify(
						event,
						null,
						2,
					)}`,
				);
			} else if (event.kind === 'task') {
				const task = event as Task;
				console.log(
					`${prefix} ${colorize('blue', 'ℹ️ Task Stream Event:')} ID: ${
						task.id
					}, Context: ${task.contextId}, Status: ${task.status.state}`,
				);
				if (task.id !== currentTaskId) {
					console.log(
						colorize(
							'dim',
							`   Task ID updated from ${currentTaskId || 'N/A'} to ${task.id}`,
						),
					);
					currentTaskId = task.id;
				}
				if (task.contextId && task.contextId !== currentContextId) {
					currentContextId = task.contextId;
				}
				if (task.status.message) {
					// Display the task message in an Assistant box
					const messageText =
						task.status.message.parts
							?.filter(part => part.kind === 'text')
							?.map(part => part.text)
							?.join(' ') || '';
					if (messageText) {
						displayAssistantMessage(messageText);
					}
				}
				if (task.artifacts && task.artifacts.length > 0) {
					console.log(
						colorize(
							'gray',
							`   Task includes ${task.artifacts.length} artifact(s).`,
						),
					);
				}
			} else {
				console.log(
					prefix,
					colorize('yellow', 'Received unknown event structure from stream:'),
					event,
				);
			}
		}
	} catch (error: any) {
		const timestamp = new Date().toLocaleTimeString();
		const prefix = colorize('red', `\n${agentName} [${timestamp}] ERROR:`);
		console.error(
			prefix,
			`Error communicating with agent:`,
			error.message || error,
		);
		if (error.code) {
			console.error(colorize('gray', `   Code: ${error.code}`));
		}
		if (error.data) {
			console.error(colorize('gray', `   Data: ${JSON.stringify(error.data)}`));
		}
		if (!(error.code || error.data) && error.stack) {
			console.error(
				colorize('gray', error.stack.split('\n').slice(1, 3).join('\n')),
			);
		}
	} finally {
		if (isInteractive) {
			rl.prompt();
		}
	}
}

// --- Main Loop ---
async function main() {
	console.log(colorize('bright', `A2A Terminal Client`));
	console.log(colorize('dim', `Base Server URL: ${baseServerUrl}`));

	// Select agent first
	selectedAgentId = await selectAgent();

	// Build the agent-specific URL (card URL for discovery)
	serverUrl = `${baseServerUrl}/agents/${selectedAgentId}/.well-known/agent-card.json`;
	// serverUrl = `${baseServerUrl}/agents/${selectedAgentId}`;
	console.log(colorize('dim', `Agent Card URL: ${serverUrl}`));

	// Initialize the client
	client = await A2AClient.fromCardUrl(serverUrl, {
		headers: authToken ? {Authorization: `Bearer ${authToken}`} : undefined,
	} as any);

	await fetchAndDisplayAgentCard(); // Fetch the card before starting the loop

	// Check if there's a message from CLI user-input flag
	if (cliArgs.userInput) {
		console.log(
			colorize('dim', `Received message from CLI: "${cliArgs.userInput}"`),
		);
		console.log(colorize('green', `Sending message automatically...`));

		// Process the input as if it was entered interactively
		await processInput(cliArgs.userInput, false);
		console.log(colorize('yellow', '\nMessage processed. Exiting...'));
		process.exit(0);
	}

	// Check if there's input from stdin (non-interactive mode)
	// Check if stdin is a TTY (interactive) or if there's data available
	if (!process.stdin.isTTY) {
		// Non-interactive mode - read from stdin
		let input = '';
		process.stdin.setEncoding('utf8');

		for await (const chunk of process.stdin) {
			input += chunk;
		}

		const trimmedInput = input.trim();
		if (trimmedInput) {
			console.log(
				colorize('dim', `Received input from stdin: "${trimmedInput}"`),
			);
			console.log(colorize('green', `Sending message automatically...`));

			// Process the input as if it was entered interactively
			await processInput(trimmedInput, false);
			console.log(colorize('yellow', '\nMessage processed. Exiting...'));
			process.exit(0);
		}
	}

	console.log(
		colorize(
			'dim',
			`No active task or context initially. Use '/new' to start a fresh session or send a message.`,
		),
	);
	console.log(
		colorize(
			'green',
			`Enter messages, or use '/new' to start a new session. '/exit' to quit.`,
		),
	);

	rl.setPrompt(colorize('cyan', `${agentName} > You: `)); // Set initial prompt
	rl.prompt();

	rl.on('line', async line => {
		const input = line.trim();

		// Keep the prompt consistent as "You:" throughout the interaction
		rl.setPrompt(colorize('cyan', `${agentName} > You: `));

		await processInput(input);
	}).on('close', () => {
		console.log(colorize('yellow', '\nExiting A2A Terminal Client. Goodbye!'));
		process.exit(0);
	});
}

// --- Start ---
main().catch(err => {
	console.error(colorize('red', 'Unhandled error in main:'), err);
	process.exit(1);
});
