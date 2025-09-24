import {Agent, tool} from '@openai/agents';
import {AgentConfiguration} from '@openai/agents-core';
import {z} from 'zod';
import {listAgents, Agent as AgentConfig} from '../api/agentsApi.js';
import {
	RepositoryContainer,
	DefaultRepositoryContainer,
} from '../services/backing/repositoryContainer.js';

// Global tool cache to avoid redundant loading and to batch per server
class ToolCache {
	private static instance: ToolCache;
	private toolsCache: Map<string, any> = new Map(); // toolId -> tool instance
	private serverToolsCache: Map<string, any[]> = new Map(); // serverId -> raw MCP tools
	private loadingPromises: Map<string, Promise<any[]>> = new Map(); // serverId -> loading promise

	static getInstance(): ToolCache {
		if (!ToolCache.instance) {
			ToolCache.instance = new ToolCache();
		}
		return ToolCache.instance;
	}

	async getTools(
		toolIds: string[],
		repositories?: RepositoryContainer,
	): Promise<any[]> {
		if (toolIds.length === 0) return [];

		const cachedTools: any[] = [];
		const uncachedToolIds: string[] = [];

		for (const toolId of toolIds as string[]) {
			const cachedTool = this.toolsCache.get(toolId);
			if (cachedTool) {
				cachedTools.push(cachedTool);
			} else {
				uncachedToolIds.push(toolId);
			}
		}

		if (uncachedToolIds.length > 0) {
			const newTools = await this.loadAndCacheTools(
				uncachedToolIds,
				repositories,
			);
			cachedTools.push(...newTools);
		}

		return cachedTools;
	}

	private async loadAndCacheTools(
		toolIds: string[],
		repositories?: RepositoryContainer,
	): Promise<any[]> {
		const toolsByServer: {[serverId: string]: string[]} = {};

		for (const toolId of toolIds) {
			const [serverId, toolName] = toolId.split('.');
			if (!serverId || !toolName) {
				console.warn(
					`⚠️  Invalid toolId format: ${toolId}. Expected format: serverId.toolName`,
				);
				continue;
			}
			if (!toolsByServer[serverId]) toolsByServer[serverId] = [];
			toolsByServer[serverId].push(toolName);
		}

		const serverLoadPromises = Object.entries(toolsByServer).map(
			([serverId, toolNames]) =>
				this.loadToolsFromServer(serverId, toolNames, repositories),
		);

		const results = await Promise.allSettled(serverLoadPromises);
		const all: any[] = [];
		for (const r of results) {
			if (r.status === 'fulfilled') all.push(...r.value);
			else console.error('Failed to load tools from server:', r.reason);
		}
		return all;
	}

	private async loadToolsFromServer(
		serverId: string,
		requestedToolNames: string[],
		repositories?: RepositoryContainer,
	): Promise<any[]> {
		// If already loading, await the same promise to dedupe
		if (!this.serverToolsCache.has(serverId)) {
			let promise = this.loadingPromises.get(serverId);
			if (!promise) {
				promise = this.fetchServerTools(serverId, repositories);
				this.loadingPromises.set(serverId, promise);
			}
			try {
				const mcpTools = await promise;
				this.serverToolsCache.set(serverId, mcpTools);
			} finally {
				this.loadingPromises.delete(serverId);
			}
		}

		const availableTools = this.serverToolsCache.get(serverId) || [];
		const tools: any[] = [];
		for (const toolName of requestedToolNames) {
			const mcpTool = availableTools.find((t: any) => t.name === toolName);
			if (!mcpTool) {
				console.warn(`⚠️  Tool ${toolName} not found in server ${serverId}`);
				continue;
			}
			const toolId = `${serverId}.${toolName}`;
			if (!this.toolsCache.has(toolId)) {
				this.toolsCache.set(
					toolId,
					this.createToolWrapper(serverId, mcpTool, repositories),
				);
			}
			tools.push(this.toolsCache.get(toolId));
		}
		return tools;
	}

	private async fetchServerTools(
		serverId: string,
		repositories?: RepositoryContainer,
	): Promise<any[]> {
		try {
			const {handleMcpServerRequest} = await import('../api/mcpServersApi.js');
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
			return listResult.result?.tools || [];
		} catch (error) {
			console.error(
				`❌ Error loading tools from MCP server ${serverId}:`,
				error,
			);
			return [];
		}
	}

	private createToolWrapper(
		serverId: string,
		mcpTool: any,
		repositories?: RepositoryContainer,
	) {
		return tool({
			name: mcpTool.name,
			description: mcpTool.description || 'No description available',
			parameters: createZodSchemaFromJsonSchema(
				mcpTool.inputSchema,
			) as z.ZodObject<any>,
			async execute(params: any) {
				return await invokeMcpTool(
					serverId,
					mcpTool.name,
					params,
					repositories,
				);
			},
			needsApproval: true,
		});
	}

	clearCache(): void {
		this.toolsCache.clear();
		this.serverToolsCache.clear();
		this.loadingPromises.clear();
	}
}

// Load agents using the agents API
async function loadAgents(
	repositories?: RepositoryContainer,
): Promise<AgentConfig[]> {
	try {
		const response = await listAgents(repositories);
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

// Optimized context builder: preload all tools for main and handoff agents
async function getContextWithOptimizedToolLoading(
	agentId: string,
	repositories?: RepositoryContainer,
) {
	const agentsById = await createAgentsLookup(repositories);
	const context = agentsById[agentId];
	if (!context) {
		throw new Error(
			`Agent with ID ${agentId} not found in agents configuration`,
		);
	}

	const allToolIds = new Set<string>();
	if (context.toolIds)
		context.toolIds.forEach((id: string) => allToolIds.add(id));

	const validHandoffConfigs: AgentConfig[] = [];
	if (context.handoffIds && Array.isArray(context.handoffIds)) {
		for (const handoffId of context.handoffIds) {
			const handoffConfig = agentsById[handoffId];
			if (handoffConfig) {
				validHandoffConfigs.push(handoffConfig);
				if (handoffConfig.toolIds)
					handoffConfig.toolIds.forEach((id: string) => allToolIds.add(id));
			} else {
				console.warn(
					`⚠️  Handoff agent with ID ${handoffId} not found in agents configuration`,
				);
			}
		}
	}

	const toolCache = ToolCache.getInstance();
	const allTools = await toolCache.getTools(
		Array.from(allToolIds),
		repositories,
	);
	// Loaded tools for agents

	// Build a lookup by toolId for precise mapping (serverId.toolName)
	const toolById = new Map<string, any>();
	for (const toolId of allToolIds) {
		const [_serverId, toolName] = toolId.split('.');
		// The cache stored wrappers keyed by toolId; resolve again for exact instance
		const cached = (toolCache as any).toolsCache?.get?.(toolId);
		if (cached) {
			toolById.set(toolId, cached);
		} else {
			// Fallback: match by name from allTools (may be ambiguous across servers)
			const byName = allTools.find(
				(t: unknown) => (t as {name?: string}).name === toolName,
			);
			if (byName) toolById.set(toolId, byName);
		}
	}

	const handoffs: Agent[] = [];
	for (const handoffConfig of validHandoffConfigs) {
		const handoffTools = (handoffConfig.toolIds || [])
			.map((toolId: string) => toolById.get(toolId))
			.filter(Boolean);
		// Assigned tools to handoff agent
		const agent = new Agent({
			name: handoffConfig.name,
			handoffDescription: handoffConfig.handoffDescription,
			instructions: handoffConfig.instructions,
			model: handoffConfig.model,
			modelSettings: handoffConfig.modelSettings,
			tools: handoffTools,
		});
		handoffs.push(agent);
	}

	const mainAgentTools = (context.toolIds || [])
		.map((toolId: string) => toolById.get(toolId))
		.filter(Boolean);

	return {
		model: context.model,
		name: context.name,
		instructions: context.instructions,
		handoffs: handoffs,
		modelSettings: context.modelSettings,
		tools: mainAgentTools,
		toolIds: context.toolIds || [],
	};
}

export class AgentFactory {
	repositories: RepositoryContainer;

	constructor(repositories?: RepositoryContainer) {
		this.repositories = repositories || new DefaultRepositoryContainer();
	}

	async buildAgentConfig(agentId: string) {
		const context = await getContextWithOptimizedToolLoading(
			agentId,
			this.repositories,
		);
		const agentConfig: AgentConfiguration = {
			name: context.name,
			instructions: context.instructions,
			handoffs: context.handoffs,
			model: context.model,
			modelSettings: context.modelSettings,
			tools: context.tools,
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
		const agent = new Agent(agentConfig);
		return agent;
	}

	clearToolCache(): void {
		ToolCache.getInstance().clearCache();
	}
}
