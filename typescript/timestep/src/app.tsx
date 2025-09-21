import React, {useState, useEffect} from 'react';
import {Text, Box} from 'ink';
import {spawn, ChildProcess} from 'child_process';
import {getTimestepPaths, loadAppConfig} from './utils.js';

type Props = {
	name: string | undefined;
	command?: string;
	flags?: {
		agentId?: string;
		autoApprove?: boolean;
		userInput?: string;
	};
};

type Agent = {
	id: string;
	name: string;
	instructions: string;
	handoffIds?: string[];
	handoffDescription?: string;
	toolIds: string[];
	model: string;
	modelSettings: {
		temperature: number;
	};
};

type Chat = {
	contextId: string;
	agentId: string;
	taskHistories: Record<string, any[]>;
	taskStates: Record<string, any>;
	tasks: any[];
};

type Model = {
	id: string;
	created: number;
	object: string;
	owned_by: string;
};

type Tool = {
	id: string;
	name: string;
	description: string;
	serverId: string;
	serverName: string;
	inputSchema: any;
	category: string;
	status: string;
};

type Trace = {
	id: string;
	object: string;
	created_at: string;
	duration_ms: number | null;
	first_5_agents: string[] | null;
	group_id: string | null;
	handoff_count: number;
	tool_count: number;
	workflow_name: string;
	metadata: Record<string, unknown>;
};

type McpServer = {
	id: string;
	name: string;
	description: string;
	serverUrl: string;
	enabled: boolean;
	authToken?: string;
};

type ModelProvider = {
	id: string;
	provider: string;
	api_key?: string;
	base_url: string;
	models_url: string;
};

export default function App({name = 'Stranger', command, flags}: Props) {
	const appConfig = loadAppConfig();
	const [agents, setAgents] = useState<Agent[]>([]);
	const [chats, setChats] = useState<Chat[]>([]);
	const [models, setModels] = useState<Model[]>([]);
	const [tools, setTools] = useState<Tool[]>([]);
	const [traces, setTraces] = useState<Trace[]>([]);
	const [mcpServers, setMcpServers] = useState<McpServer[]>([]);
	const [modelProviders, setModelProviders] = useState<ModelProvider[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [serverStarted, setServerStarted] = useState(false);
	const [serverStopped, setServerStopped] = useState(false);
	const [chatStarted, setChatStarted] = useState(false);
	const [availableAgents, setAvailableAgents] = useState<any[]>([]);
	const [chatProcess, setChatProcess] = useState<ChildProcess | null>(null);

	useEffect(() => {
		if (command === 'list-agents') {
			fetchAgents();
		} else if (command === 'list-chats') {
			fetchChats();
		} else if (command === 'list-mcp-servers') {
			fetchMcpServers();
		} else if (command === 'list-model-providers') {
			fetchModelProviders();
		} else if (command === 'list-models') {
			fetchModels();
		} else if (command === 'list-tools') {
			fetchTools();
		} else if (command === 'list-traces') {
			fetchTraces();
		} else if (command === 'server') {
			startServer();
		} else if (command === 'stop') {
			stopServer();
		} else if (command === 'chat') {
			startChat();
		} else if (command === 'ag-ui-chat') {
			startAgUiChat();
		}
	}, [command]);

	// Cleanup chat process on unmount
	useEffect(() => {
		return () => {
			if (chatProcess) {
				chatProcess.kill();
			}
		};
	}, [chatProcess]);

	// Helper function to get available agents
	const getAvailableAgents = async (): Promise<{
		success: boolean;
		agents?: any[];
		error?: string;
	}> => {
		try {
			const {listAgents} = await import('./api/agentsApi.js');
			const response = await listAgents();
			return {success: true, agents: response.data};
		} catch (error) {
			return {
				success: false,
				error: error instanceof Error ? error.message : 'Failed to load agents',
			};
		}
	};

	// Helper function to start chat process
	const startChatProcess = async (
		options: {
			agentId?: string;
			autoApprove?: boolean;
			userInput?: string;
		} = {},
	): Promise<{success: boolean; error?: string}> => {
		try {
			const timestepPaths = getTimestepPaths();

			// Build arguments for the a2aClient
			const args = ['src/a2aClient.ts'];

			if (options.agentId) {
				args.push('--agentId', options.agentId);
			}

			if (options.autoApprove) {
				args.push('--auto-approve');
			}

			if (options.userInput) {
				args.push('--user-input', options.userInput);
			}

			// Check if agents config exists
			const fs = await import('node:fs');
			if (!fs.existsSync(timestepPaths.agentsConfig)) {
				return {
					success: false,
					error: `Agents configuration not found at: ${timestepPaths.agentsConfig}`,
				};
			}

			// Start the a2aClient process using tsx (TypeScript runner)
			const childProcess = spawn('npx', ['tsx', ...args], {
				stdio: 'inherit',
				cwd: process.cwd(),
			});

			setChatProcess(childProcess);

			return new Promise(resolve => {
				childProcess.on('error', (error: Error) => {
					resolve({
						success: false,
						error: `Failed to start A2A client: ${error.message}`,
					});
				});

				childProcess.on('spawn', () => {
					resolve({success: true});
				});

				childProcess.on('exit', (code: number | null) => {
					if (code !== 0) {
						resolve({
							success: false,
							error: `A2A client exited with code ${code}`,
						});
					}
				});
			});
		} catch (error) {
			return {
				success: false,
				error: error instanceof Error ? error.message : 'Unknown error',
			};
		}
	};

	// Helper function to start AG-UI chat process
	const startAgUiChatProcess = async (
		options: {
			agentId?: string;
			userInput?: string;
		} = {},
	): Promise<{success: boolean; error?: string}> => {
		try {
			const timestepPaths = getTimestepPaths();

			// Build arguments for the agUiClient
			const args = ['src/agUiClient.ts'];

			if (options.agentId) {
				args.push('--agentId', options.agentId);
			}

			if (options.userInput) {
				args.push('--user-input', options.userInput);
			}

			// Check if agents config exists
			const fs = await import('node:fs');
			if (!fs.existsSync(timestepPaths.agentsConfig)) {
				return {
					success: false,
					error: `Agents configuration not found at: ${timestepPaths.agentsConfig}`,
				};
			}

			// Start the agUiClient process using tsx (TypeScript runner)
			const childProcess = spawn('npx', ['tsx', ...args], {
				stdio: 'inherit',
				cwd: process.cwd(),
			});

			setChatProcess(childProcess);

			return new Promise(resolve => {
				childProcess.on('error', (error: Error) => {
					resolve({
						success: false,
						error: `Failed to start AG-UI client: ${error.message}`,
					});
				});

				childProcess.on('spawn', () => {
					resolve({success: true});
				});

				childProcess.on('exit', (code: number | null) => {
					if (code !== 0) {
						resolve({
							success: false,
							error: `AG-UI client exited with code ${code}`,
						});
					}
				});
			});
		} catch (error) {
			return {
				success: false,
				error: error instanceof Error ? error.message : 'Unknown error',
			};
		}
	};

	const startServer = () => {
		try {
			// Check if Node.js is available
			const nodeCheck = spawn('node', ['--version'], {stdio: 'pipe'});

			nodeCheck.on('error', () => {
				setError(
					'Node.js is not installed. Please install Node.js from https://nodejs.org/',
				);
			});

			nodeCheck.on('close', code => {
				if (code === 0) {
					// Start the server from compiled dist/server.js
					const server = spawn('node', ['dist/server.js'], {
						stdio: 'inherit',
						detached: true,
					});

					server.unref(); // Allow the parent process to exit
					setServerStarted(true);
				} else {
					setError(
						'Node.js is not installed. Please install Node.js from https://nodejs.org/',
					);
				}
			});
		} catch (_err) {
			setError('Failed to start server');
		}
	};

	const stopServer = () => {
		setLoading(true);
		setError(null);
		setServerStopped(false);

		try {
			const ports = [appConfig.appPort!];
			let processesKilled = 0;

			ports.forEach(port => {
				// Kill process on each port
				const killProcess = spawn('lsof', ['-ti', `:${port}`], {stdio: 'pipe'});

				killProcess.stdout.on('data', data => {
					const pid = data.toString().trim();
					if (pid) {
						const killCmd = spawn('kill', ['-9', pid], {stdio: 'pipe'});
						killCmd.on('close', code => {
							if (code === 0) {
								processesKilled++;
								console.log(`✅ Killed process ${pid} on port ${port}`);
							}
						});
					}
				});

				killProcess.on('close', () => {
					// Check if all ports have been processed
					if (processesKilled >= 0) {
						// Allow for some ports having no processes
						setServerStopped(true);
						setLoading(false);
					}
				});
			});

			// Set timeout to complete the operation
			setTimeout(() => {
				setServerStopped(true);
				setLoading(false);
			}, 2000);
		} catch (_err) {
			setError('Failed to stop server');
			setLoading(false);
		}
	};

	const startChat = async () => {
		setLoading(true);
		setError(null);

		try {
			// First, check if agents are available
			const agentsResult = await getAvailableAgents();

			if (!agentsResult.success) {
				setError(agentsResult.error || 'Failed to load agents');
				setLoading(false);
				return;
			}

			setAvailableAgents(agentsResult.agents || []);

			// Start the chat with the first available agent or let user choose
			const result = await startChatProcess({
				agentId: flags?.agentId,
				autoApprove: flags?.autoApprove || false,
				userInput: flags?.userInput,
			});

			if (result.success) {
				setChatStarted(true);
			} else {
				setError(result.error || 'Failed to start chat');
			}
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to start chat');
		} finally {
			setLoading(false);
		}
	};

	const startAgUiChat = async () => {
		setLoading(true);
		setError(null);

		try {
			// First, check if agents are available
			const agentsResult = await getAvailableAgents();

			if (!agentsResult.success) {
				setError(agentsResult.error || 'Failed to load agents');
				setLoading(false);
				return;
			}

			setAvailableAgents(agentsResult.agents || []);

			// Start the AG-UI chat
			const result = await startAgUiChatProcess({
				agentId: flags?.agentId,
				userInput: flags?.userInput,
			});

			if (result.success) {
				setChatStarted(true);
			} else {
				setError(result.error || 'Failed to start AG-UI chat');
			}
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to start AG-UI chat');
		} finally {
			setLoading(false);
		}
	};

	const fetchAgents = async () => {
		setLoading(true);
		setError(null);

		try {
			const response = await fetch(
				`http://localhost:${appConfig.appPort}/agents`,
			);
			if (!response.ok) {
				throw new Error(`Server responded with ${response.status}`);
			}
			const data = await response.json();
			setAgents(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to fetch agents');
		} finally {
			setLoading(false);
		}
	};

	const fetchChats = async () => {
		setLoading(true);
		setError(null);

		try {
			const response = await fetch(
				`http://localhost:${appConfig.appPort}/chats`,
			);
			if (!response.ok) {
				throw new Error(`Server responded with ${response.status}`);
			}
			const data = await response.json();
			setChats(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to fetch chats');
		} finally {
			setLoading(false);
		}
	};

	const fetchModels = async () => {
		setLoading(true);
		setError(null);

		try {
			const response = await fetch(
				`http://localhost:${appConfig.appPort}/models`,
			);
			if (!response.ok) {
				throw new Error(`Server responded with ${response.status}`);
			}
			const data = await response.json();
			setModels(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to fetch models');
		} finally {
			setLoading(false);
		}
	};

	const fetchTools = async () => {
		setLoading(true);
		setError(null);

		try {
			const response = await fetch(
				`http://localhost:${appConfig.appPort}/tools`,
			);
			if (!response.ok) {
				throw new Error(`Server responded with ${response.status}`);
			}
			const data = await response.json();
			setTools(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to fetch tools');
		} finally {
			setLoading(false);
		}
	};

	const fetchTraces = async () => {
		setLoading(true);
		setError(null);

		try {
			const response = await fetch(
				`http://localhost:${appConfig.appPort}/traces`,
			);
			if (!response.ok) {
				throw new Error(`Server responded with ${response.status}`);
			}
			const data = await response.json();
			setTraces(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to fetch traces');
		} finally {
			setLoading(false);
		}
	};

	const fetchMcpServers = async () => {
		setLoading(true);
		setError(null);

		try {
			const response = await fetch(
				`http://localhost:${appConfig.appPort}/mcp_servers`,
			);
			if (!response.ok) {
				throw new Error(`Server responded with ${response.status}`);
			}
			const data = await response.json();
			setMcpServers(data);
		} catch (err) {
			setError(
				err instanceof Error ? err.message : 'Failed to fetch MCP servers',
			);
		} finally {
			setLoading(false);
		}
	};

	const fetchModelProviders = async () => {
		setLoading(true);
		setError(null);

		try {
			const response = await fetch(
				`http://localhost:${appConfig.appPort}/model_providers`,
			);
			if (!response.ok) {
				throw new Error(`Server responded with ${response.status}`);
			}
			const data = await response.json();
			setModelProviders(data);
		} catch (err) {
			setError(
				err instanceof Error ? err.message : 'Failed to fetch model providers',
			);
		} finally {
			setLoading(false);
		}
	};

	if (command === 'server') {
		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
				</Box>
			);
		}

		if (serverStarted) {
			return (
				<Box flexDirection="column">
					<Text color="green">✅ Timestep Agents Server started!</Text>
					<Text>
						CLI server is available at: http://localhost:{appConfig.appPort}
					</Text>
					<Text>
						Agents endpoint: http://localhost:{appConfig.appPort}/agents
					</Text>
					<Text color="blue">Use 'timestep list-agents' to fetch agents</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="green">🚀 Starting Timestep Agents Server...</Text>
				<Text>Checking for Node.js...</Text>
			</Box>
		);
	}

	if (command === 'stop') {
		if (loading) {
			return (
				<Box flexDirection="column">
					<Text color="blue">🛑 Stopping Timestep servers...</Text>
					<Text>Checking port {appConfig.appPort}...</Text>
				</Box>
			);
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
				</Box>
			);
		}

		if (serverStopped) {
			return (
				<Box flexDirection="column">
					<Text color="green">✅ Timestep servers stopped successfully!</Text>
					<Text>Process on port {appConfig.appPort} has been terminated.</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="blue">🛑 Preparing to stop servers...</Text>
			</Box>
		);
	}

	if (command === 'chat') {
		if (loading) {
			return <Text color="blue">Loading agents and starting chat...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure your timestep configuration is set up correctly.
					</Text>
					<Text color="gray">Expected config at:</Text>
					{availableAgents.length === 0 && (
						<Text color="gray">
							{' '}
							- Run 'timestep list-agents' to see if agents are configured
						</Text>
					)}
				</Box>
			);
		}

		if (chatStarted) {
			return (
				<Box flexDirection="column">
					<Text color="green">🚀 Interactive chat session started!</Text>
					<Text color="blue">
						You can now chat with the agent. Use Ctrl+C to exit.
					</Text>
					{availableAgents.length > 0 && (
						<Box flexDirection="column" marginTop={1}>
							<Text color="cyan">Available agents:</Text>
							{availableAgents.slice(0, 3).map((agent, index) => (
								<Text key={index} color="gray">
									{' '}
									• {agent.name}
								</Text>
							))}
							{availableAgents.length > 3 && (
								<Text color="gray">
									{' '}
									• ... and {availableAgents.length - 3} more
								</Text>
							)}
						</Box>
					)}
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="blue">🤖 Preparing to start chat...</Text>
				<Text>Loading available agents...</Text>
			</Box>
		);
	}

	if (command === 'ag-ui-chat') {
		if (loading) {
			return <Text color="blue">Loading agents and starting AG-UI chat...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure your timestep configuration is set up correctly.
					</Text>
					<Text color="gray">Expected config at:</Text>
					{availableAgents.length === 0 && (
						<Text color="gray">
							{' '}
							- Run 'timestep list-agents' to see if agents are configured
						</Text>
					)}
				</Box>
			);
		}

		if (chatStarted) {
			return (
				<Box flexDirection="column">
					<Text color="green">🚀 Interactive AG-UI chat session started!</Text>
					<Text color="blue">
						You can now chat with the agent using AG-UI protocol. Use Ctrl+C to exit.
					</Text>
					{availableAgents.length > 0 && (
						<Box flexDirection="column" marginTop={1}>
							<Text color="cyan">Available agents:</Text>
							{availableAgents.slice(0, 3).map((agent, index) => (
								<Text key={index} color="gray">
									{' '}
									• {agent.name}
								</Text>
							))}
							{availableAgents.length > 3 && (
								<Text color="gray">
									{' '}
									• ... and {availableAgents.length - 3} more
								</Text>
							)}
						</Box>
					)}
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="blue">🤖 Preparing to start AG-UI chat...</Text>
				<Text>Loading available agents...</Text>
			</Box>
		);
	}

	if (command === 'list-agents') {
		if (loading) {
			return <Text color="blue">Loading agents...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure the server is running with: timestep server
					</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="green">🤖 Available Agents:</Text>
				{agents.map(agent => (
					<Box key={agent.id} flexDirection="column" marginLeft={2}>
						<Text>
							<Text color="blue">•</Text> <Text color="cyan">{agent.name}</Text>
							<Text color="gray"> (ID: {agent.id.slice(0, 8)}...)</Text>
						</Text>
						<Box marginLeft={2}>
							<Text color="gray">
								Model: {agent.model} | Temperature:{' '}
								{agent.modelSettings.temperature}
							</Text>
						</Box>
						<Box marginLeft={2}>
							<Text color="gray">Tools: {agent.toolIds.length} available</Text>
						</Box>
						{agent.handoffDescription && (
							<Box marginLeft={2}>
								<Text color="gray">{agent.handoffDescription}</Text>
							</Box>
						)}
						{agent.handoffIds && agent.handoffIds.length > 0 && (
							<Box marginLeft={2}>
								<Text color="gray">
									Can handoff to: {agent.handoffIds.length} other agents
								</Text>
							</Box>
						)}
					</Box>
				))}
			</Box>
		);
	}

	if (command === 'list-chats') {
		if (loading) {
			return <Text color="blue">Loading chats...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure the server is running with: timestep server
					</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="green">💬 Available Chats:</Text>
				{chats.map(chat => (
					<Box key={chat.contextId} flexDirection="column" marginLeft={2}>
						<Text>
							<Text color="blue">•</Text>{' '}
							<Text color="cyan">Context {chat.contextId.slice(0, 8)}...</Text>
						</Text>
						<Box marginLeft={2}>
							<Text color="gray">Agent: {chat.agentId}</Text>
						</Box>
						<Box marginLeft={2}>
							<Text color="gray">
								Tasks: {chat.tasks.length} | Conversations:{' '}
								{Object.keys(chat.taskHistories).length}
							</Text>
						</Box>
					</Box>
				))}
			</Box>
		);
	}

	if (command === 'list-models') {
		if (loading) {
			return <Text color="blue">Loading models...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure the server is running with: timestep server
					</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="green">🤖 Available Models:</Text>
				{models.map(model => (
					<Box key={model.id} flexDirection="column" marginLeft={2}>
						<Text>
							<Text color="blue">•</Text> <Text color="cyan">{model.id}</Text>
						</Text>
						<Box marginLeft={2}>
							<Text color="gray">
								Owner: {model.owned_by} | Created:{' '}
								{new Date(model.created * 1000).toLocaleDateString()}
							</Text>
						</Box>
					</Box>
				))}
			</Box>
		);
	}

	if (command === 'list-tools') {
		if (loading) {
			return <Text color="blue">Loading tools...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure the server is running with: timestep server
					</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="green">🔧 Available Tools:</Text>
				{tools.map(tool => (
					<Box key={tool.id} marginLeft={2}>
						<Text>
							<Text color="blue">•</Text> <Text color="cyan">{tool.name}</Text>
							<Text color="gray"> ({tool.serverName})</Text>
						</Text>
					</Box>
				))}
			</Box>
		);
	}

	if (command === 'list-traces') {
		if (loading) {
			return <Text color="blue">Loading traces...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure the server is running with: timestep server
					</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="green">📊 Recent Traces:</Text>
				{traces.map(trace => (
					<Box key={trace.id} flexDirection="column" marginLeft={2}>
						<Text>
							<Text color="blue">•</Text>{' '}
							<Text color="cyan">{trace.workflow_name}</Text>
							<Text color="gray"> ({trace.id})</Text>
						</Text>
						<Box marginLeft={2}>
							<Text color="gray">
								Agents: {trace.first_5_agents?.join(', ') || 'None'} | Tools:{' '}
								{trace.tool_count} | Handoffs: {trace.handoff_count}
							</Text>
						</Box>
						<Box marginLeft={2}>
							<Text color="gray">
								Duration:{' '}
								{trace.duration_ms ? `${trace.duration_ms}ms` : 'Unknown'} |
								Created: {new Date(trace.created_at).toLocaleString()}
							</Text>
						</Box>
					</Box>
				))}
			</Box>
		);
	}

	if (command === 'list-mcp-servers') {
		if (loading) {
			return <Text color="blue">Loading MCP servers...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure the server is running with: timestep server
					</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="green">🖥️ Configured MCP Servers:</Text>
				{mcpServers.map(server => (
					<Box key={server.id} flexDirection="column" marginLeft={2}>
						<Text>
							<Text color="blue">•</Text>{' '}
							<Text color="cyan">{server.name}</Text>
							<Text color="gray"> ({server.id})</Text>
						</Text>
						<Box marginLeft={2}>
							<Text color="gray">
								Description: {server.description || 'N/A'}
							</Text>
						</Box>
						<Box marginLeft={2}>
							<Text color="gray">
								URL: {server.serverUrl || 'N/A'} | Status:{' '}
								<Text color={server.enabled ? 'green' : 'yellow'}>
									{server.enabled ? 'enabled' : 'disabled'}
								</Text>
							</Text>
						</Box>
						{server.authToken && (
							<Box marginLeft={2}>
								<Text color="gray">
									Auth Token: {server.authToken ? '***' : 'N/A'}
								</Text>
							</Box>
						)}
					</Box>
				))}
			</Box>
		);
	}

	if (command === 'list-model-providers') {
		if (loading) {
			return <Text color="blue">Loading model providers...</Text>;
		}

		if (error) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {error}</Text>
					<Text color="yellow">
						Make sure the server is running with: timestep server
					</Text>
				</Box>
			);
		}

		return (
			<Box flexDirection="column">
				<Text color="green">🤖 Configured Model Providers:</Text>
				{modelProviders.map(provider => (
					<Box key={provider.id} flexDirection="column" marginLeft={2}>
						<Text>
							<Text color="blue">•</Text>{' '}
							<Text color="cyan">{provider.provider}</Text>
							<Text color="gray"> ({provider.id})</Text>
						</Text>
						<Box marginLeft={2}>
							<Text color="gray">Base URL: {provider.base_url || 'N/A'}</Text>
						</Box>
						<Box marginLeft={2}>
							<Text color="gray">
								Models URL: {provider.models_url || 'N/A'}
							</Text>
						</Box>
						{provider.api_key && (
							<Box marginLeft={2}>
								<Text color="gray">
									API Key: {provider.api_key ? '***' : 'N/A'}
								</Text>
							</Box>
						)}
					</Box>
				))}
			</Box>
		);
	}

	if (command === 'get-version') {
		const [versionInfo, setVersionInfo] = useState<{
			version: string;
			name: string;
			description: string;
			timestamp: string;
		} | null>(null);
		const [versionError, setVersionError] = useState<string | null>(null);

		useEffect(() => {
			const fetchVersion = async () => {
				try {
					const {getVersion} = await import('./utils.js');
					const info = await getVersion();
					setVersionInfo(info);
				} catch (error) {
					setVersionError(
						error instanceof Error ? error.message : 'Failed to get version',
					);
				}
			};

			fetchVersion();
		}, []);

		if (versionError) {
			return (
				<Box flexDirection="column">
					<Text color="red">❌ Error: {versionError}</Text>
				</Box>
			);
		}

		if (!versionInfo) {
			return <Text color="blue">Loading version information...</Text>;
		}

		return (
			<Box flexDirection="column">
				<Text color="cyan">📦 Timestep Version Information</Text>
				<Box marginLeft={2} flexDirection="column">
					<Text>
						<Text color="blue">Name:</Text>{' '}
						<Text color="white">{versionInfo.name}</Text>
					</Text>
					<Text>
						<Text color="blue">Version:</Text>{' '}
						<Text color="green">{versionInfo.version}</Text>
					</Text>
					<Text>
						<Text color="blue">Description:</Text>{' '}
						<Text color="white">{versionInfo.description}</Text>
					</Text>
					<Text>
						<Text color="blue">Timestamp:</Text>{' '}
						<Text color="gray">{versionInfo.timestamp}</Text>
					</Text>
				</Box>
			</Box>
		);
	}

	// Default greeting
	return (
		<Text>
			Hello, <Text color="green">{name}</Text>
		</Text>
	);
}
