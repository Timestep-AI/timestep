import {Agent, tool} from '@openai/agents';
import {AgentConfiguration} from '@openai/agents-core';
import {z} from 'zod';
import {listAgents, Agent as AgentConfig} from '../api/agentsApi.js';
import {
	RepositoryContainer,
	DefaultRepositoryContainer,
} from '../services/backing/repositoryContainer.js';

// Load agents using the agents API
async function loadAgents(
	repositories?: RepositoryContainer,
): Promise<AgentConfig[]> {
	try {
		const response = await listAgents(repositories);
		// console.log(`📋 Loaded ${response.data.length} agents from agents API`);
		return response.data;
	} catch (error) {
		console.error(`Error loading agents from API: ${error}`);
		throw new Error(
			`Unable to load agents: ${
				error instanceof Error ? error.message : 'Unknown error'
			}`,
		);
	}
}

// MCP servers are now loaded dynamically via API calls

// Create agent lookup map by ID
async function createAgentsLookup(
	repositories?: RepositoryContainer,
): Promise<{[id: string]: AgentConfig}> {
	const agents = await loadAgents(repositories);
	const agentsById: {[id: string]: AgentConfig} = {};

	for (const agent of agents) {
		agentsById[agent.id] = agent;
	}

	return agentsById;
}

// Function to invoke a tool through the MCP server API endpoint
async function invokeMcpTool(
	serverId: string,
	toolName: string,
	parameters: any,
	repositories?: RepositoryContainer,
): Promise<string> {
	try {
		const {handleMcpServerRequest} = await import('../api/mcpServersApi.js');

		// Convert A2A SDK tool names back to MCP server expected names
		// A2A SDK converts hyphens to underscores, but MCP server expects hyphens
		const mcpToolName = toolName.replace(/_/g, '-');

		const request = {
			jsonrpc: '2.0',
			method: 'tools/call',
			params: {
				name: mcpToolName,
				arguments: parameters,
			},
			id: Math.random().toString(36).substring(7),
		};

		const result = await handleMcpServerRequest(
			serverId,
			request,
			repositories,
		);

		// Extract text content from the result
		let content = '';
		if (result.result?.content && Array.isArray(result.result.content)) {
			for (const item of result.result.content) {
				if (item.type === 'text' && typeof item.text === 'string') {
					content += item.text;
				}
			}
		}

		return content || `Tool ${toolName} executed successfully`;
	} catch (error) {
		console.error(`Error invoking MCP tool ${toolName}:`, error);
		return `Error executing tool ${toolName}: ${
			error instanceof Error ? error.message : String(error)
		}`;
	}
}

// Function to load specific tools based on toolIds using the MCP server API endpoints
async function loadToolsForAgent(
	toolIds: string[],
	repositories?: RepositoryContainer,
): Promise<any[]> {
	const tools = [];
	const {handleMcpServerRequest} = await import('../api/mcpServersApi.js');

	// Group tools by MCP server
	const serverIds = new Set<string>();
	const toolsByServer: {[serverId: string]: string[]} = {};

	for (const toolId of toolIds) {
		const [serverId, toolName] = toolId.split('.');
		if (!serverId || !toolName) {
			console.warn(
				`⚠️  Invalid toolId format: ${toolId}. Expected format: serverId.toolName`,
			);
			continue;
		}

		serverIds.add(serverId);
		if (!toolsByServer[serverId]) {
			toolsByServer[serverId] = [];
		}
		toolsByServer[serverId].push(toolName);
	}

	// Load tools from each server using the API endpoints
	for (const serverId of serverIds) {
		try {
			// Call the tools/list endpoint through the API
			const listRequest = {
				jsonrpc: '2.0',
				method: 'tools/list',
				id: Math.random().toString(36).substring(7),
			};

			const listResult = await handleMcpServerRequest(
				serverId,
				listRequest,
				repositories,
			);
			const availableTools = listResult.result?.tools || [];

			// Filter to only include requested tools
			const requestedToolNames = toolsByServer[serverId];
			const mcpTools = availableTools.filter((mcpTool: any) =>
				requestedToolNames.includes(mcpTool.name),
			);

			// Create tool wrappers for each available tool
			for (const mcpTool of mcpTools) {
				const dynamicTool = tool({
					name: mcpTool.name,
					description: mcpTool.description || 'No description available',
					parameters: createZodSchemaFromJsonSchema(
						mcpTool.inputSchema,
					) as z.ZodObject<any>,
					async execute(params: any) {
						const result = await invokeMcpTool(
							serverId,
							mcpTool.name,
							params,
							repositories,
						);
						return result;
					},
					needsApproval: true,
				});

				tools.push(dynamicTool);
			}
		} catch (error) {
			console.error(
				`❌ Error loading tools from MCP server ${serverId}:`,
				error,
			);
			console.error(
				`❌ Error details:`,
				error instanceof Error ? error.message : String(error),
			);
			console.error(
				`❌ Error stack:`,
				error instanceof Error ? error.stack : 'No stack trace',
			);
		}
	}

	console.log(
		`✅ Loaded ${tools.length} tools from ${serverIds.size} MCP servers`,
	);
	return tools;
}

// Helper function to create Zod schema from JSON schema
function createZodSchemaFromJsonSchema(jsonSchema: any): z.ZodSchema {
	if (!jsonSchema || jsonSchema.type !== 'object') {
		return z.object({});
	}

	const shape: Record<string, z.ZodSchema> = {};

	if (jsonSchema.properties) {
		for (const [key, prop] of Object.entries(jsonSchema.properties)) {
			const propSchema = prop as any;
			switch (propSchema.type) {
				case 'string':
					shape[key] = z.string();
					break;
				case 'number':
					shape[key] = z.number();
					break;
				case 'boolean':
					shape[key] = z.boolean();
					break;
				case 'array':
					shape[key] = z.array(z.any());
					break;
				default:
					shape[key] = z.any();
			}
		}
	}

	return z.object(shape);
}

async function getContext(agentId: string, repositories?: RepositoryContainer) {
	// Get the agents lookup map
	const agentsById = await createAgentsLookup(repositories);

	// Get the specific agent configuration by ID
	const context = agentsById[agentId];
	if (!context) {
		throw new Error(
			`Agent with ID ${agentId} not found in agents configuration`,
		);
	}

	const handoffs: Agent[] = [];

	// Resolve handoff agents by their IDs and load their tools
	if (context.handoffIds && Array.isArray(context.handoffIds)) {
		for (const handoffId of context.handoffIds) {
			const handoffConfig = agentsById[handoffId];
			if (handoffConfig) {
				// Load tools for this handoff agent based on its toolIds
				const handoffTools = await loadToolsForAgent(
					handoffConfig.toolIds || [],
					repositories,
				);
				console.log(
					`🔄 Loaded ${handoffTools.length} tools for handoff agent ${handoffConfig.name}`,
				);

				const agent = new Agent({
					name: handoffConfig.name,
					handoffDescription: handoffConfig.handoffDescription,
					instructions: handoffConfig.instructions,
					model: handoffConfig.model,
					modelSettings: handoffConfig.modelSettings,
					tools: handoffTools, // ✅ Include tools for handoff agent
				});
				handoffs.push(agent);
			} else {
				console.warn(
					`⚠️  Handoff agent with ID ${handoffId} not found in agents configuration`,
				);
			}
		}
	}

	return {
		model: context.model,
		name: context.name,
		instructions: context.instructions,
		handoffs: handoffs,
		modelSettings: context.modelSettings,
		toolIds: context.toolIds || [], // Include toolIds from agent configuration
	};
}

export class AgentFactory {
	repositories: RepositoryContainer;

	constructor(repositories?: RepositoryContainer) {
		this.repositories = repositories || new DefaultRepositoryContainer();
	}

	async buildAgentConfig(agentId: string) {
		const context = await getContext(agentId, this.repositories);

		// Load tools based on agent's toolIds
		const tools = await loadToolsForAgent(
			context.toolIds || [],
			this.repositories,
		);

		const agentConfig: AgentConfiguration = {
			name: context.name,
			instructions: context.instructions,
			handoffs: context.handoffs,
			model: context.model,
			modelSettings: context.modelSettings,
			tools: tools,
			handoffDescription: '',
			mcpServers: [],
			inputGuardrails: [],
			outputGuardrails: [],
			outputType: 'text',
			toolUseBehavior: 'run_llm_again',
			resetToolChoice: true,
		};

		return {
			config: agentConfig,
			createAgent: () => this.createAgent(agentConfig),
		};
	}

	createAgent(agentConfig: AgentConfiguration) {
		// return new Agent(agentConfig);
		const agent = new Agent(agentConfig);
		return agent;
	}
}
