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

// --- State ---
let currentTaskId: string | undefined = undefined; // Initialize as undefined
let currentContextId: string | undefined = undefined; // Initialize as undefined
let pendingToolCalls: ToolCall[] = []; // Track pending tool calls awaiting approval
let isWaitingForApproval = false; // Track if we're waiting for user approval
let selectedAgentId: string | undefined = undefined; // Track selected agent ID
let lastDisplayedMessage: string = ''; // Track last displayed message content for delta handling
let isCurrentlyStreaming: boolean = false; // Track if we're currently in streaming mode
let streamedTaskIds: Set<string> = new Set(); // Track which tasks we've streamed content for
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
			console.log(`📋 Loaded ${agents.length} agents from /agents endpoint`);

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

		// Only show status line for non-streaming updates or final updates
		const isStreaming = state === 'working' && !update.final;
		if (!isStreaming || update.final) {
			console.log(
				`${prefix} ${stateEmoji} Status: ${colorize(
					stateColor,
					state,
				)} (Task: ${update.taskId}, Context: ${update.contextId}) ${
					update.final ? colorize('bright', '[FINAL]') : ''
				}`,
			);
		}

		// Clear task ID when task is final and completed (not just working or input-required)
		if (update.final && state === 'completed') {
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

		if (update.status.message) {
			// Detect if this is streaming mode (working state and not final)
			const isStreaming = state === 'working' && !update.final;

			if (isStreaming) {
				// For streaming mode, first show the status line if it's the first chunk
				if (lastDisplayedMessage === '') {
					// This is the first streaming chunk, show the initial status prefix
					console.log(
						`${prefix} ${stateEmoji} Status: ${colorize(
							stateColor,
							state,
						)} (Task: ${update.taskId}, Context: ${update.contextId})`,
					);
					process.stdout.write(colorize('green', `     📝 `)); // Show prefix without newline
					isCurrentlyStreaming = true;
					streamedTaskIds.add(update.taskId); // Mark this task as having streamed content
				}
				printMessageContentStreaming(update.status.message, true);
			} else {
				// For non-streaming mode, check if we were just streaming
				if (isCurrentlyStreaming && state === 'completed') {
					// We were streaming and now we're done - replace the streamed content with clean final version
					// Move to a new line after streaming and show clean final message
					console.log('\n'); // Move to new line after streaming
					// Don't show the message content again since it was already streamed
					isCurrentlyStreaming = false;
				} else {
					// Regular non-streaming message
					printMessageContentStreaming(update.status.message, false);
				}
			}

			// Check if this is a tool call approval request
			if (state === 'input-required' && update.status.message.parts) {
				for (const part of update.status.message.parts) {
					if (
						part.kind === 'text' &&
						part.text.includes('Human approval required for tool execution:')
					) {
						await handleToolCallFromStatusMessage(
							part.text,
							update.taskId,
							update.contextId,
						);
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
			// Check kind property
			console.log(colorize('green', `     📝 ${part.text}`));
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
			// Check kind property
			const dataPart = part as DataPart;
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
				// For streaming, only show the delta (new content)
				const currentText = part.text;
				if (
					currentText.length > lastDisplayedMessage.length &&
					currentText.startsWith(lastDisplayedMessage)
				) {
					// Show only the new content
					const delta = currentText.slice(lastDisplayedMessage.length);
					if (delta) {
						// Use process.stdout.write for streaming without newlines
						process.stdout.write(colorize('green', delta));
					}
					lastDisplayedMessage = currentText;
				} else if (currentText !== lastDisplayedMessage) {
					// Content changed completely, show with prefix
					console.log(colorize('green', `     📝 ${currentText}`));
					lastDisplayedMessage = currentText;
				}
			} else {
				// For non-streaming, show complete message
				console.log(colorize('green', `     📝 ${part.text}`));
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
	console.log(colorize('cyan', `\n🔧 Tool Call Approval Required:`));
	console.log(
		colorize('bright', `  Name: ${artifact.name || 'Unnamed Tool Call'}`),
	);
	if (artifact.description) {
		console.log(colorize('dim', `  Description: ${artifact.description}`));
	}

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
			console.log(
				colorize(
					'green',
					`\n  🤖 Auto-${decision} enabled for this tool (${reason})`,
				),
			);

			// Check if we have an active task before proceeding with auto-approval
			if (!currentTaskId) {
				// Silently ignore tool call artifacts for completed tasks
				return;
			}

			// Set approval state and tool call data
			pendingToolCalls = [toolCallData];
			isWaitingForApproval = true;

			// Automatically handle the tool call
			await handleToolApproval(decision, reason);
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

	console.log(colorize('yellow', `\n  💡 Tool Call Options:`));
	console.log(
		colorize('dim', `     • approve [reason] - Execute the tool call as-is`),
	);
	console.log(colorize('dim', `     • reject [reason] - Cancel the tool call`));
	console.log(
		colorize(
			'dim',
			`     • modify <param>=<value> - Change a parameter (e.g., 'modify city=San Francisco')`,
		),
	);
	console.log(
		colorize(
			'dim',
			`     • auto-approve - Approve and remember for future calls to this tool`,
		),
	);
	console.log(
		colorize(
			'dim',
			`     • auto-reject - Reject and remember for future calls to this tool`,
		),
	);
	console.log(
		colorize(
			'dim',
			`     • show-params - Display current parameters for modification`,
		),
	);
}

// --- Tool Call Parsing from Status Message ---
async function handleToolCallFromStatusMessage(
	messageText: string,
	_taskId: string,
	_contextId: string,
): Promise<void> {
	// Parse tool call information from the status message
	// Format: "Human approval required for tool execution: get_weather({\"city\":\"Oakland\"})"
	const toolCallMatch = messageText.match(
		/Human approval required for tool execution: (.+)/,
	);
	if (!toolCallMatch) {
		console.log(
			colorize(
				'red',
				'Could not parse tool call information from status message',
			),
		);
		return;
	}

	const toolCallText = toolCallMatch[1];
	console.log(colorize('cyan', `\n🔧 Tool Call Approval Required:`));
	console.log(
		colorize('bright', `  Parsed from status message: ${toolCallText}`),
	);

	// Parse the tool call (simple regex-based parsing)
	const toolMatch = toolCallText.match(/^(\w+)\((.+)\)$/);
	if (!toolMatch) {
		console.log(colorize('red', 'Could not parse tool call format'));
		return;
	}

	const toolName = toolMatch[1];
	const parametersText = toolMatch[2];

	let parameters: Record<string, any> = {};
	try {
		parameters = JSON.parse(parametersText);
	} catch (error) {
		console.log(
			colorize('red', 'Could not parse tool call parameters as JSON'),
		);
		return;
	}

	console.log(
		colorize('yellow', `\n  🛠️  Tool: ${colorize('bright', toolName)}`),
	);
	console.log(colorize('dim', `     ID: call_${Date.now()}`));

	if (Object.keys(parameters).length > 0) {
		console.log(colorize('dim', `     Parameters:`));
		Object.entries(parameters).forEach(([key, value]) => {
			console.log(colorize('dim', `       ${key}: ${JSON.stringify(value)}`));
		});
	}

	console.log(colorize('red', `\n  🔐 Approval Required:`));
	console.log(colorize('dim', `     Type: approval`));
	console.log(
		colorize('dim', `     Reason: Tool execution requires human approval`),
	);

	// Store the tool call for approval handling
	// Note: We're generating a new ID here since we don't have access to the original tool call ID
	// In a real implementation, we'd need to capture this from the artifact events
	const toolCall: ToolCall = {
		id: `call_${Date.now()}`,
		name: toolName,
		parameters: parameters,
		artifactId: `tool-call-${Date.now()}`,
	};

	pendingToolCalls.push(toolCall);
	isWaitingForApproval = true;

	// Check for auto-preferences
	const autoPreference = checkAutoPreference(toolName);
	if (autoPreference || cliArgs.autoApprove) {
		const reason = cliArgs.autoApprove
			? 'CLI auto-approve flag'
			: `Auto-${autoPreference} based on saved preference`;
		const decision = (cliArgs.autoApprove ? 'approve' : autoPreference!) as
			| 'approve'
			| 'reject';
		console.log(
			colorize(
				'green',
				`\n  🤖 Auto-${decision} enabled for this tool (${reason})`,
			),
		);
		// Automatically handle the tool call
		await handleToolApproval(decision, reason);
		return;
	}

	console.log(colorize('yellow', `\n  💡 Tool Call Options:`));
	console.log(
		colorize('dim', `     • approve [reason] - Execute the tool call as-is`),
	);
	console.log(colorize('dim', `     • reject [reason] - Cancel the tool call`));
	console.log(
		colorize(
			'dim',
			`     • modify <param>=<value> - Change a parameter (e.g., 'modify city=San Francisco')`,
		),
	);
	console.log(
		colorize(
			'dim',
			`     • auto-approve - Approve and remember for future calls to this tool`,
		),
	);
	console.log(
		colorize(
			'dim',
			`     • auto-reject - Reject and remember for future calls to this tool`,
		),
	);
	console.log(
		colorize(
			'dim',
			`     • show-params - Display current parameters for modification`,
		),
	);
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
): Promise<void> {
	if (pendingToolCalls.length === 0) {
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
	const toolCall = pendingToolCalls[0]; // Handle the first pending tool call

	console.log(
		colorize('cyan', `\n🔧 Processing tool call decision: ${decision}`),
	);

	let toolResult: string;

	if (isApproved) {
		console.log(colorize('green', `✅ Approving tool call: ${toolCall.name}`));
		if (reason) {
			console.log(colorize('dim', `   Reason: ${reason}`));
		}

		// Execute the tool call
		toolResult = await executeToolCall(toolCall);
		console.log(colorize('green', `   Tool result: ${toolResult}`));
	} else {
		console.log(colorize('red', `❌ Rejecting tool call: ${toolCall.name}`));
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
						callId: toolCall.id,
						artifactId: toolCall.artifactId, // Reference to original tool call artifact
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
		console.log(
			colorize(
				'cyan',
				`\n📤 Sending tool response to continue conversation...`,
			),
		);
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
				console.log(colorize('blue', `\n📋 Task Update: ${task.status.state}`));

				// For completed tasks, skip the message content since it was already streamed
				if (task.status.message) {
					if (task.status.state === 'completed') {
						// Skip showing the message content for completed tasks - it was streamed already
						// Just show a simple note that the task completed
					} else {
						printMessageContent(task.status.message);
					}
				}
			}
		}

		// Clear the specific tool call that was just processed
		const processedToolCallId = pendingToolCalls[0]?.id;

		// Remove the processed tool call but keep any newer ones that may have been added
		pendingToolCalls = pendingToolCalls.filter(
			tc => tc.id !== processedToolCallId,
		);

		// Only reset approval state if no more tool calls are pending
		if (pendingToolCalls.length === 0) {
			isWaitingForApproval = false;
		} else {
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
		console.log(colorize('red', 'Sending message...'));
		console.log(colorize('dim', `   Current state before sending:`));
		console.log(
			colorize('dim', `     currentTaskId: ${currentTaskId || 'undefined'}`),
		);
		console.log(
			colorize(
				'dim',
				`     currentContextId: ${currentContextId || 'undefined'}`,
			),
		);
		console.log(
			colorize(
				'dim',
				`     messagePayload.taskId: ${messagePayload.taskId || 'undefined'}`,
			),
		);
		console.log(
			colorize(
				'dim',
				`     messagePayload.contextId: ${
					messagePayload.contextId || 'undefined'
				}`,
			),
		);

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
					console.log(colorize('gray', '   Task includes message:'));
					printMessageContent(task.status.message);
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
