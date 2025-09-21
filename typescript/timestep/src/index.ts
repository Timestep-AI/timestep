// Library exports for npm module usage

// Core functions
export {getTimestepPaths, getVersion} from './utils.js';
export {
	encryptSecret,
	decryptSecret,
	isEncryptedSecret,
	maskSecret,
	getCurrentUserId,
} from './utils.js';
export {listModels} from './api/modelsApi.js';
export {listContexts} from './api/contextsApi.js';
export {listTraces} from './api/tracesApi.js';
export {listTools} from './api/toolsApi.js';

// Agent functions
export {
	listAgents,
	getAgent,
	isAgentAvailable,
	handleAgentRequest,
	handleListAgents,
} from './api/agentsApi.js';

// MCP Server functions
export {
	listMcpServers,
	getMcpServer,
	handleMcpServerRequest,
} from './api/mcpServersApi.js';

// Model Provider functions
export {listModelProviders, getModelProvider} from './api/modelProvidersApi.js';

// Core classes and types
export {Context} from './types/context.js';
export {DefaultRepositoryContainer} from './services/backing/repositoryContainer.js';
export {TimestepAIAgentExecutor} from './core/agentExecutor.js';
export {ContextAwareRequestHandler} from './api/contextAwareRequestHandler.js';

// Configuration functions
export {
	getDefaultMcpServers,
	getBuiltinMcpServer,
} from './config/defaultMcpServers.js';
export {getDefaultAgents} from './config/defaultAgents.js';
export {getDefaultModelProviders} from './config/defaultModelProviders.js';

// TypeScript types
export type {Agent, ListAgentsResponse} from './api/agentsApi.js';
export type {McpServer, ListMcpServersResponse} from './api/mcpServersApi.js';
export type {
	ModelProvider,
	ListModelProvidersResponse,
} from './api/modelProvidersApi.js';
export type {Trace} from './api/tracesApi.js';
export type {Tool} from './api/toolsApi.js';
export type {Repository} from './services/backing/repository.js';
export type {RepositoryContainer} from './services/backing/repositoryContainer.js';

// MCP SDK re-exports for convenience
export {Server} from '@modelcontextprotocol/sdk/server/index.js';
export {StreamableHTTPServerTransport} from '@modelcontextprotocol/sdk/server/streamableHttp.js';
export {
	CallToolRequestSchema,
	ListToolsRequestSchema,
	ListResourcesRequestSchema,
	ReadResourceRequestSchema,
	ListPromptsRequestSchema,
	GetPromptRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

export type {Tool as McpTool} from '@modelcontextprotocol/sdk/types.js';
