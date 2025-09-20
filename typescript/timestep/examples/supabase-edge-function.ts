/**
 * Supabase Edge Function for Timestep Server
 *
 * This is a complete, self-contained Timestep server for Supabase Edge Functions.
 * It uses Supabase database repositories since Edge Functions can't access the filesystem.
 *
 * To set up this function:
 * 1. Run the SQL schema (provided at bottom) in your Supabase SQL editor
 * 2. Copy this entire file to supabase/functions/YOUR_FUNCTION_NAME/index.ts
 * 3. Deploy with: supabase functions deploy YOUR_FUNCTION_NAME
 *
 * That's it! The function automatically detects its name from the URL.
 * Access endpoints like: https://YOUR_PROJECT.supabase.co/functions/v1/YOUR_FUNCTION_NAME/agents
 */

import 'https://deno.land/x/xhr@0.1.0/mod.ts';
import {createClient} from 'https://esm.sh/@supabase/supabase-js@2';
// Import everything from timestep library (includes MCP SDK re-exports)
import {
	Context,
	TimestepAIAgentExecutor,
	getModelProvider,
	maskSecret,
	getVersion,
	handleAgentRequest,
	listAgents,
	listContexts,
	listMcpServers,
	listModelProviders,
	listModels,
	listTools,
	listTraces,
	// Types
	type Agent,
	type McpServer,
	type ModelProvider,
	type Repository,
	type RepositoryContainer,
} from 'npm:@timestep-ai/timestep@2025.9.201303';

/**
 * Supabase Agent Repository Implementation
 */
class SupabaseAgentRepository implements Repository<Agent, string> {
	constructor(private supabase: any, private userId: string | null) {}

	async list(): Promise<Agent[]> {
		if (!this.userId) return [];
		// Upsert defaults for this user
		try {
			const {getDefaultAgents} = await import(
				'npm:@timestep-ai/timestep@2025.9.201303'
			);
			const defaultAgents = getDefaultAgents();
			for (const agent of defaultAgents) {
				await this.save(agent);
			}
		} catch (saveError) {
			console.warn(`Failed to upsert default agents: ${saveError}`);
		}

		const {data, error} = await this.supabase
			.from('agents')
			.select('*')
			.eq('user_id', this.userId);
		if (error) throw new Error(`Failed to list agents: ${error.message}`);
		return data || [];
	}

	async load(id: string): Promise<Agent | null> {
		const {data, error} = await this.supabase
			.from('agents')
			.select('*')
			.eq('id', id)
			.single();

		if (error && error.code !== 'PGRST116') {
			throw new Error(`Failed to load agent: ${error.message}`);
		}
		return data || null;
	}

	async save(agent: Agent): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		const toSave: any = {
			id: (agent as any).id,
			user_id: this.userId,
			name: (agent as any).name,
			instructions: (agent as any).instructions,
			handoff_description: (agent as any).handoffDescription ?? null,
			handoff_ids: (agent as any).handoffIds ?? [],
			tool_ids: (agent as any).toolIds ?? [],
			model: (agent as any).model,
			model_settings: (agent as any).modelSettings ?? {},
		};
		const {error} = await this.supabase
			.from('agents')
			.upsert([toSave], {onConflict: 'user_id,id'});
		if (error) throw new Error(`Failed to save agent: ${error.message}`);
	}

	async delete(id: string): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		const {error} = await this.supabase
			.from('agents')
			.delete()
			.eq('user_id', this.userId)
			.eq('id', id);

		if (error) throw new Error(`Failed to delete agent: ${error.message}`);
	}

	async exists(id: string): Promise<boolean> {
		const agent = await this.load(id);
		return agent !== null;
	}

	async getOrCreate(id: string, ...createArgs: any[]): Promise<Agent> {
		const existing = await this.load(id);
		if (existing) {
			return existing;
		}

		// For demonstration - in real implementation, you'd create a proper agent
		// based on createArgs or throw an error for agents since they shouldn't be auto-created
		throw new Error(
			'Auto-creation of agents not supported - please create agents explicitly',
		);
	}
}

/**
 * Supabase Context Repository Implementation
 */
class SupabaseContextRepository implements Repository<Context, string> {
	constructor(private supabase: any, private userId: string | null) {}

	async list(): Promise<Context[]> {
		if (!this.userId) return [];
		const {data, error} = await this.supabase
			.from('contexts')
			.select('*')
			.eq('user_id', this.userId);

		if (error) throw new Error(`Failed to list contexts: ${error.message}`);
		return (data || []).map((item: any) => {
			const context = new Context(item.context_id, item.agent_id);
			context.taskHistories = item.task_histories || {};
			context.taskStates = item.task_states || {};
			context.tasks = item.tasks || [];
			return context;
		});
	}

	async load(id: string): Promise<Context | null> {
		if (!this.userId) return null;
		const {data, error} = await this.supabase
			.from('contexts')
			.select('*')
			.eq('user_id', this.userId)
			.eq('context_id', id)
			.single();

		if (error && error.code !== 'PGRST116') {
			throw new Error(`Failed to load context: ${error.message}`);
		}
		if (!data) return null;

		const context = new Context(data.context_id, data.agent_id);
		context.taskHistories = data.task_histories || {};
		context.taskStates = data.task_states || {};
		context.tasks = data.tasks || [];
		return context;
	}

	async save(context: Context): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		const {error} = await this.supabase.from('contexts').upsert([
			{
				id: crypto.randomUUID(),
				user_id: this.userId,
				context_id: context.contextId,
				agent_id: context.agentId,
				task_histories: context.taskHistories,
				created_at: new Date().toISOString(),
			},
		]);

		if (error) throw new Error(`Failed to save context: ${error.message}`);
	}

	async delete(id: string): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		const {error} = await this.supabase
			.from('contexts')
			.delete()
			.eq('user_id', this.userId)
			.eq('context_id', id);

		if (error) throw new Error(`Failed to delete context: ${error.message}`);
	}

	async exists(id: string): Promise<boolean> {
		const context = await this.load(id);
		return context !== null;
	}

	async getOrCreate(contextId: string, agentId: string): Promise<Context> {
		const existing = await this.load(contextId);
		if (existing) {
			return existing;
		}

		const newContext = new Context(contextId, agentId);
		await this.save(newContext);
		return newContext;
	}
}

/**
 * Supabase MCP Server Repository Implementation
 */
class SupabaseMcpServerRepository implements Repository<McpServer, string> {
	constructor(
		private supabase: any,
		private baseUrl: string | undefined,
		private userId: string | null,
	) {}

	async list(): Promise<McpServer[]> {
		if (!this.userId) return [];
		// Always upsert defaults for this user
		try {
			const {getDefaultMcpServers} = await import(
				'npm:@timestep-ai/timestep@2025.9.201303'
			);
			const defaults = getDefaultMcpServers(this.baseUrl);
			for (const server of defaults) {
				await this.save(server);
			}
		} catch (e) {
			console.warn(`Failed to upsert default MCP servers: ${e}`);
		}

		const {data, error} = await this.supabase
			.from('mcp_servers')
			.select('*')
			.eq('user_id', this.userId);
		if (error) throw new Error(`Failed to list MCP servers: ${error.message}`);

		const servers = (data || []).map((row: any) => {
			return {
				id: row.id,
				name: row.name,
				description: row.description ?? row.name,
				serverUrl: row.server_url ?? '',
				enabled: row.enabled ?? true,
				authToken: row.auth_token,
			} as McpServer;
		});

		if (servers.length === 0) {
			const {getDefaultMcpServers} = await import(
				'npm:@timestep-ai/timestep@2025.9.201303'
			);
			const defaultServers = getDefaultMcpServers(this.baseUrl);
			try {
				for (const server of defaultServers) {
					await this.save(server);
				}
				console.log(
					`🔌 Created ${defaultServers.length} default MCP servers in database`,
				);
			} catch (saveError) {
				console.warn(
					`Failed to save default MCP servers to database: ${saveError}`,
				);
			}
			return defaultServers;
		}

		return servers;
	}

	async load(id: string): Promise<McpServer | null> {
		const {data, error} = await this.supabase
			.from('mcp_servers')
			.select('*')
			.eq('id', id)
			.single();

		if (error && error.code !== 'PGRST116') {
			throw new Error(`Failed to load MCP server: ${error.message}`);
		}

		if (!data) return null;

		// Apply the same transformation logic as the list method
		return {
			id: data.id,
			name: data.name,
			description: data.description ?? data.name,
			serverUrl: data.server_url ?? '',
			enabled: data.enabled ?? true,
			authToken: data.auth_token,
		} as McpServer;
	}

	async save(server: McpServer): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		// Persist server using snake_case column mapping
		const toSave: any = {
			id: server.id,
			user_id: this.userId,
			name: server.name,
			description: (server as any).description ?? server.name,
			server_url: (server as any).serverUrl,
			enabled: server.enabled,
			auth_token: (server as any).authToken,
		};
		const {isEncryptedSecret, encryptSecret} = await import(
			'npm:@timestep-ai/timestep@2025.9.201303'
		);

		// Encrypt auth token if provided and not already encrypted
		if ((server as any).authToken !== undefined) {
			let token = (server as any).authToken as string | undefined;
			if (token && !isEncryptedSecret(token)) {
				try {
					token = await encryptSecret(token);
				} catch {}
			}
			toSave.auth_token = token ?? null;
		}
		const {error} = await this.supabase
			.from('mcp_servers')
			.upsert([toSave], {onConflict: 'user_id,id'});
		if (error) throw new Error(`Failed to save MCP server: ${error.message}`);
	}

	async delete(id: string): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		const {error} = await this.supabase
			.from('mcp_servers')
			.delete()
			.eq('user_id', this.userId)
			.eq('id', id);
		if (error) throw new Error(`Failed to delete MCP server: ${error.message}`);
	}

	async exists(id: string): Promise<boolean> {
		const server = await this.load(id);
		return server !== null;
	}

	async getOrCreate(id: string, ...createArgs: any[]): Promise<McpServer> {
		const existing = await this.load(id);
		if (existing) return existing;
		throw new Error(
			'Auto-creation of MCP servers not supported - please create servers explicitly',
		);
	}
}

class SupabaseModelProviderRepository
	implements Repository<ModelProvider, string>
{
	constructor(private supabase: any, private userId: string | null) {}

	async list(): Promise<ModelProvider[]> {
		if (!this.userId) return [];
		// Always upsert defaults for this user
		try {
			const {getDefaultModelProviders} = await import(
				'npm:@timestep-ai/timestep@2025.9.201303'
			);
			const defaults = getDefaultModelProviders();
			for (const p of defaults) {
				await this.save(p);
			}
		} catch (e) {
			console.warn(`Failed to upsert default model providers: ${e}`);
		}

		const {data, error} = await this.supabase
			.from('model_providers')
			.select('*')
			.eq('user_id', this.userId);

		if (error)
			throw new Error(`Failed to list model providers: ${error.message}`);

		const providers = (data || []).map((row: any) => ({
			id: row.id,
			provider: row.provider,
			apiKey: row.api_key,
			baseUrl: row.base_url,
			modelsUrl: row.models_url,
		}));

		return providers;
	}

	async load(id: string): Promise<ModelProvider | null> {
		const {data, error} = await this.supabase
			.from('model_providers')
			.select('*')
			.eq('id', id)
			.single();

		if (error && error.code !== 'PGRST116') {
			throw new Error(`Failed to load model provider: ${error.message}`);
		}
		return data || null;
	}

	async save(provider: ModelProvider): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		// Map to snake_case; encrypt apiKey if provided
		const toSave: any = {
			id: provider.id,
			user_id: this.userId,
			provider: provider.provider,
			base_url: (provider as any).baseUrl ?? (provider as any).base_url,
			models_url: (provider as any).modelsUrl ?? (provider as any).models_url,
		};
		const {isEncryptedSecret, encryptSecret} = await import(
			'npm:@timestep-ai/timestep@2025.9.201303'
		);
		if ((provider as any).apiKey !== undefined) {
			let key = (provider as any).apiKey as string | undefined;
			if (key && !isEncryptedSecret(key)) {
				try {
					key = await encryptSecret(key);
				} catch {}
			}
			toSave.api_key = key ?? null;
		}
		const {error} = await this.supabase
			.from('model_providers')
			.upsert([toSave], {onConflict: 'user_id,id'});

		if (error)
			throw new Error(`Failed to save model provider: ${error.message}`);
	}

	async delete(id: string): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		const {error} = await this.supabase
			.from('model_providers')
			.delete()
			.eq('user_id', this.userId)
			.eq('id', id);

		if (error)
			throw new Error(`Failed to delete model provider: ${error.message}`);
	}

	async exists(id: string): Promise<boolean> {
		const provider = await this.load(id);
		return provider !== null;
	}

	async getOrCreate(id: string, ...createArgs: any[]): Promise<ModelProvider> {
		const existing = await this.load(id);
		if (existing) {
			return existing;
		}

		// For demonstration - in real implementation, you'd create a proper model provider
		// based on createArgs or throw an error since they usually shouldn't be auto-created
		throw new Error(
			'Auto-creation of model providers not supported - please create providers explicitly',
		);
	}
}

/** End reordering: McpServer before ModelProvider to match alphabetical by class name */

/**
 * Supabase Repository Container Implementation
 */
class SupabaseRepositoryContainer implements RepositoryContainer {
	constructor(
		private supabase: any,
		private baseUrl: string | undefined,
		private userId: string | null,
	) {}

	get agents() {
		return new SupabaseAgentRepository(this.supabase, this.userId);
	}
	get contexts() {
		return new SupabaseContextRepository(this.supabase, this.userId);
	}
	get modelProviders() {
		return new SupabaseModelProviderRepository(this.supabase, this.userId);
	}
	get mcpServers() {
		return new SupabaseMcpServerRepository(
			this.supabase,
			this.baseUrl,
			this.userId,
		);
	}
}

/**
 * Custom task store for Supabase environment
 */
class SupabaseTaskStore {
	private store: Map<string, any> = new Map();

	async load(taskId: string): Promise<any | undefined> {
		console.log(`📋 SupabaseTaskStore.load(${taskId})`);
		const entry = this.store.get(taskId);
		if (entry) {
			console.log(`📋 SupabaseTaskStore.load(${taskId}) -> FOUND`);
			return {...entry};
		} else {
			console.log(`📋 SupabaseTaskStore.load(${taskId}) -> NOT FOUND`);
			return undefined;
		}
	}

	async save(task: any): Promise<void> {
		console.log(`📋 SupabaseTaskStore.save(${task.id})`);
		this.store.set(task.id, {...task});
		console.log(`📋 SupabaseTaskStore.save(${task.id}) -> SAVED`);
	}
}

// Initialize Supabase client
const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
// Prefer service role key (bypasses RLS) for server-side operations; fallback to anon key if not set
const supabaseServiceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
const supabaseKey =
	supabaseServiceRoleKey || Deno.env.get('SUPABASE_ANON_KEY')!;
const supabase = createClient(supabaseUrl, supabaseKey);

// Task store and server config
const taskStore = new SupabaseTaskStore();

// Configure the port from environment or default
const port = parseInt(Deno.env.get('PORT') || '3000');

console.log('🦕 Starting Timestep Server with Custom Supabase Repositories');
console.log(`🌐 Server will run on port ${port}`);

// Start the server with custom repositories
Deno.serve({port}, async (request: Request) => {
	const url = new URL(request.url);

	// Extract the path after the Supabase function name first
	// Supabase Edge Functions receive URLs like /server/agents (not /functions/v1/server/agents)
	// The /functions/v1/ part is already stripped by Supabase before reaching our code
	const pathParts = url.pathname.split('/');

	// For Edge Functions, the URL pattern is: /FUNCTION_NAME/api_path
	// So we need to extract everything after the function name (index 1)
	let cleanPath = '/';
	let functionName = 'unknown';

	if (pathParts.length > 1 && pathParts[1]) {
		functionName = pathParts[1]; // First part is the function name
		const apiParts = pathParts.slice(2); // Everything after the function name
		cleanPath = apiParts.length > 0 ? '/' + apiParts.join('/') : '/';
	}

	// Generate base URL for MCP servers from the current request
	const baseUrl = `${url.protocol}//${url.host}/${functionName}`;

	// Derive user from Authorization header (Bearer JWT)
	const authHeader = request.headers.get('Authorization') || '';
	const jwt = authHeader.startsWith('Bearer ') ? authHeader.substring(7) : null;
	let userId: string | null = null;
	if (jwt) {
		try {
			const {
				data: {user},
				error,
			} = await supabase.auth.getUser(jwt);
			if (!error && user) userId = user.id as string;
		} catch (_e) {
			// ignore; userId stays null
		}
	}

	// Create repositories and executor per request so user scoping is correct
	const repositories = new SupabaseRepositoryContainer(
		supabase,
		baseUrl,
		userId,
	);
	const agentExecutor = new TimestepAIAgentExecutor({
		repositories,
	});
	console.log(
		`🔧 Using base URL: ${baseUrl} | userId: ${userId ?? 'anonymous'}`,
	);

	// Remove trailing slash except for root
	cleanPath = cleanPath === '/' ? '/' : cleanPath.replace(/\/$/, '');

	console.log(
		`🔍 Request: ${request.method} ${url.pathname} -> mapped to: ${cleanPath}`,
	);

	const headers = {
		'Access-Control-Allow-Origin': '*',
		'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
		'Access-Control-Allow-Headers':
			'authorization, x-client-info, apikey, content-type',
		'Content-Type': 'application/json',
		'X-Runtime': 'Supabase-Edge-Function-Custom-Repositories',
		'X-Deployment-ID': Deno.env.get('DENO_DEPLOYMENT_ID') || 'local',
	};

	if (request.method === 'OPTIONS') {
		return new Response(null, {status: 200, headers});
	}

	try {
		// Enforce auth for all non-public endpoints
		const isPublic =
			cleanPath === '/' ||
			cleanPath === '' ||
			cleanPath === '/version' ||
			cleanPath === '/health' ||
			cleanPath === '/supabase-health';
		if (!userId && !isPublic) {
			return new Response(
				JSON.stringify({
					error: 'Unauthorized',
					message: 'Missing or invalid Authorization header',
					path: cleanPath,
				}),
				{status: 401, headers},
			);
		}

		// Root endpoint - useful for debugging path mapping
		if (cleanPath === '/' || cleanPath === '') {
			return new Response(
				JSON.stringify({
					message: 'Timestep Server is running',
					runtime: 'Supabase Edge Function with Custom Repositories',
					detectedFunctionName: functionName,
					originalPath: url.pathname,
					mappedPath: cleanPath,
					availableEndpoints: [
						'/agents',
						'/agents/{agentId}',
						'/chats',
						'/health',
						'/mcp_servers',
						'/mcp_servers/{serverId}',
						'/models',
						'/model_providers',
						'/model_providers/{providerId}',
						'/tools',
						'/traces',
						'/version',
					],
				}),
				{status: 200, headers},
			);
		}

		// Version endpoint - returns timestep package version info
		if (cleanPath === '/version') {
			try {
				const versionInfo = await getVersion();
				return new Response(
					JSON.stringify({
						...versionInfo,
						runtime: 'Supabase Edge Function with Custom Repositories',
					}),
					{status: 200, headers},
				);
			} catch (error) {
				return new Response(
					JSON.stringify({
						error:
							error instanceof Error
								? error.message
								: 'Failed to read version information',
					}),
					{status: 500, headers},
				);
			}
		}

		// Health check endpoints
		if (cleanPath === '/health' || cleanPath === '/supabase-health') {
			return new Response(
				JSON.stringify({
					status: 'healthy',
					runtime: 'Supabase Edge Function with Custom Repositories',
					timestamp: new Date().toISOString(),
					denoVersion: Deno.version.deno,
					deploymentId: Deno.env.get('DENO_DEPLOYMENT_ID') || 'local',
					region: Deno.env.get('DENO_REGION') || 'unknown',
					path: cleanPath,
					repositories: [
						'agents',
						'contexts',
						'model_providers',
						'mcp_servers',
					],
				}),
				{status: 200, headers},
			);
		}

		// API endpoints using custom repositories
		if (cleanPath === '/agents') {
			const result = await listAgents(repositories);
			return new Response(JSON.stringify(result.data), {status: 200, headers});
		}

		if (cleanPath === '/chats') {
			const result = await listContexts(repositories);
			return new Response(JSON.stringify(result.data), {status: 200, headers});
		}

		if (cleanPath === '/mcp_servers') {
			const result = await listMcpServers(repositories);
			const masked = result.data.map((s: any) => ({
				id: s.id,
				name: s.name,
				description: s.description,
				serverUrl: s.serverUrl,
				enabled: s.enabled,
				hasAuthToken: !!s.authToken,
				maskedAuthToken: maskSecret(s.authToken),
			}));
			return new Response(JSON.stringify(masked), {status: 200, headers});
		}

		if (cleanPath === '/model_providers') {
			const result = await listModelProviders(repositories);
			const masked = result.data.map((p: any) => ({
				id: p.id,
				provider: p.provider,
				baseUrl: p.baseUrl ?? p.base_url,
				modelsUrl: p.modelsUrl ?? p.models_url,
				hasApiKey: !!(p.apiKey ?? p.api_key),
				maskedApiKey: maskSecret(p.apiKey ?? p.api_key),
			}));
			return new Response(JSON.stringify(masked), {status: 200, headers});
		}

		// Get or update a specific model provider by ID
		const modelProviderMatch = cleanPath.match(/^\/model_providers\/([^\/]+)$/);
		if (modelProviderMatch) {
			const providerId = modelProviderMatch[1];
			if (request.method === 'PUT') {
				try {
					const body = await request.json().catch(() => ({}));
					const provider = {
						...(body || {}),
						id: providerId,
					} as ModelProvider;
					await repositories.modelProviders.save(provider);
					return new Response(JSON.stringify(provider), {
						status: 200,
						headers: {...headers, 'Content-Type': 'application/json'},
					});
				} catch (error) {
					return new Response(
						JSON.stringify({
							error: 'Failed to save model provider',
							message: error instanceof Error ? error.message : 'Unknown error',
							providerId,
						}),
						{status: 500, headers},
					);
				}
			}

			if (request.method === 'GET') {
				try {
					const provider = await getModelProvider(providerId, repositories);
					if (!provider) {
						return new Response(
							JSON.stringify({
								error: `Model provider ${providerId} not found`,
								providerId,
							}),
							{status: 404, headers},
						);
					}
					const responseBody = {
						id: provider.id,
						provider: provider.provider,
						baseUrl: (provider as any).baseUrl ?? (provider as any).base_url,
						modelsUrl:
							(provider as any).modelsUrl ?? (provider as any).models_url,
						hasApiKey: !!(
							(provider as any).apiKey ?? (provider as any).api_key
						),
						maskedApiKey: maskSecret(
							(provider as any).apiKey ?? (provider as any).api_key,
						),
					};
					return new Response(JSON.stringify(responseBody), {
						status: 200,
						headers: {...headers, 'Content-Type': 'application/json'},
					});
				} catch (error) {
					return new Response(
						JSON.stringify({
							error: 'Internal server error',
							message: error instanceof Error ? error.message : 'Unknown error',
							providerId,
						}),
						{status: 500, headers},
					);
				}
			}
		}

		if (cleanPath === '/models') {
			const result = await listModels(repositories);
			return new Response(JSON.stringify(result.data), {status: 200, headers});
		}

		if (cleanPath === '/tools') {
			const result = await listTools(repositories);
			return new Response(JSON.stringify(result.data), {status: 200, headers});
		}

		// Handle individual tool requests (GET /tools/{toolId})
		const toolMatch = cleanPath.match(/^\/tools\/(.+)$/);
		if (toolMatch && request.method === 'GET') {
			const toolId = toolMatch[1];

			try {
				// Parse toolId to extract serverId and tool name
				// Format: {serverId}.{toolName}
				const parts = toolId.split('.');
				if (parts.length !== 2) {
					return new Response(
						JSON.stringify({
							error: 'Invalid tool ID format. Expected: {serverId}.{toolName}',
							toolId: toolId,
						}),
						{status: 400, headers},
					);
				}

				const [serverId, toolName] = parts;

				// Get tool information from the MCP server
				const {handleMcpServerRequest} = await import(
					'npm:@timestep-ai/timestep@2025.9.201303'
				);

				// First, get the list of tools from the server
				const listRequest = {
					jsonrpc: '2.0',
					method: 'tools/list',
					id: 'get-tool-info',
				};

				const listResponse = await handleMcpServerRequest(
					serverId,
					listRequest,
					repositories,
				);

				if (listResponse.error) {
					return new Response(
						JSON.stringify({
							error: `Failed to list tools from server ${serverId}`,
							details: listResponse.error,
						}),
						{status: 500, headers},
					);
				}

				// Find the specific tool
				const tools = listResponse.result?.tools || [];
				const tool = tools.find((t: any) => t.name === toolName);

				if (!tool) {
					return new Response(
						JSON.stringify({
							error: `Tool '${toolName}' not found in server ${serverId}`,
							toolId: toolId,
							serverId: serverId,
							toolName: toolName,
							availableTools: tools.map((t: any) => t.name),
						}),
						{status: 404, headers},
					);
				}

				// Return tool information
				const toolInfo = {
					id: toolId,
					name: tool.name,
					description: tool.description || 'No description available',
					serverId: serverId,
					inputSchema: tool.inputSchema,
					status: 'available',
				};

				return new Response(JSON.stringify(toolInfo), {
					status: 200,
					headers: {...headers, 'Content-Type': 'application/json'},
				});
			} catch (error) {
				console.error(`Error getting tool info for ${toolId}:`, error);
				return new Response(
					JSON.stringify({
						error: 'Internal server error',
						message: error instanceof Error ? error.message : 'Unknown error',
						toolId: toolId,
					}),
					{status: 500, headers},
				);
			}
		}

		if (cleanPath === '/traces') {
			const result = await listTraces();
			return new Response(JSON.stringify(result.data), {status: 200, headers});
		}

		// Handle tool invocation (POST /tools/{toolId}/call)
		const toolCallMatch = cleanPath.match(/^\/tools\/(.+)\/call$/);
		if (toolCallMatch && request.method === 'POST') {
			const toolId = toolCallMatch[1];
			try {
				const body = await request.json().catch(() => ({}));
				const args = body?.arguments || {};
				const id = body?.id || 'tools-call';

				const parts = toolId.split('.');
				if (parts.length !== 2) {
					return new Response(
						JSON.stringify({
							jsonrpc: '2.0',
							error: {
								code: -32602,
								message:
									'Invalid toolId format. Expected {serverId}.{toolName}',
							},
							id,
						}),
						{status: 400, headers},
					);
				}

				const [serverId, toolName] = parts;
				const {handleMcpServerRequest} = await import(
					'npm:@timestep-ai/timestep@2025.9.201303'
				);

				const result = await handleMcpServerRequest(
					serverId,
					{
						jsonrpc: '2.0',
						method: 'tools/call',
						params: {name: toolName, arguments: args},
						id,
					},
					repositories,
				);

				return new Response(JSON.stringify(result), {
					status: 200,
					headers: {...headers, 'Content-Type': 'application/json'},
				});
			} catch (error) {
				return new Response(
					JSON.stringify({
						jsonrpc: '2.0',
						error: {
							code: -32603,
							message:
								error instanceof Error
									? error.message
									: 'Internal server error',
						},
						id: null,
					}),
					{status: 500, headers},
				);
			}
		}

		// Handle MCP server routes
		const mcpServerMatch = cleanPath.match(/^\/mcp_servers\/(.+)$/);
		if (mcpServerMatch) {
			const serverId = mcpServerMatch[1];

			try {
				const {handleMcpServerRequest} = await import(
					'npm:@timestep-ai/timestep@2025.9.201303'
				);

				if (request.method === 'POST') {
					const body = await request.json().catch(() => ({}));
					const result = await handleMcpServerRequest(
						serverId,
						body,
						repositories,
					);
					return new Response(JSON.stringify(result), {
						status: 200,
						headers: {...headers, 'Content-Type': 'application/json'},
					});
				}

				if (request.method === 'PUT') {
					try {
						const body = await request.json().catch(() => ({}));
						const server = {
							...(body || {}),
							id: serverId,
						} as McpServer;
						await repositories.mcpServers.save(server);
						return new Response(JSON.stringify(server), {
							status: 200,
							headers: {...headers, 'Content-Type': 'application/json'},
						});
					} catch (error) {
						return new Response(
							JSON.stringify({
								error: 'Failed to save MCP server',
								message:
									error instanceof Error ? error.message : 'Unknown error',
								serverId,
							}),
							{status: 500, headers},
						);
					}
				}

				// GET request - return full MCP server record
				const {getMcpServer} = await import(
					'npm:@timestep-ai/timestep@2025.9.201303'
				);
				const server = await getMcpServer(serverId, repositories);

				if (!server) {
					return new Response(
						JSON.stringify({
							error: `MCP server ${serverId} not found`,
						}),
						{status: 404, headers},
					);
				}

				const responseBody = {
					id: server.id,
					name: server.name,
					description: (server as any).description,
					serverUrl: (server as any).serverUrl,
					enabled: (server as any).enabled,
					hasAuthToken: !!(server as any).authToken,
					maskedAuthToken: maskSecret((server as any).authToken),
				};

				return new Response(JSON.stringify(responseBody), {
					status: 200,
					headers: {...headers, 'Content-Type': 'application/json'},
				});
			} catch (error) {
				console.error(
					`Error handling MCP server request for ${serverId}:`,
					error,
				);
				return new Response(
					JSON.stringify({
						jsonrpc: '2.0',
						error: {
							code: -32603,
							message:
								error instanceof Error
									? error.message
									: 'Internal server error',
						},
						id: null,
					}),
					{
						status: 500,
						headers: {...headers, 'Content-Type': 'application/json'},
					},
				);
			}
		}

		// Handle dynamic agent routes with custom repository
		const agentMatch = cleanPath.match(/^\/agents\/([^\/]+)(?:\/.*)?$/);
		if (agentMatch) {
			// Create a mock Express-style request object that satisfies the Request interface
			const mockReq = {
				method: request.method,
				path: cleanPath,
				originalUrl: cleanPath + url.search,
				params: {agentId: agentMatch[1]},
				body:
					request.method !== 'GET'
						? await request.json().catch(() => ({}))
						: {},
				headers: Object.fromEntries(Array.from(request.headers.entries())),
				// Add required Express Request methods as stubs
				get: (name: string) => request.headers.get(name),
				header: (name: string) => request.headers.get(name),
				accepts: () => false,
				acceptsCharsets: () => false,
				acceptsEncodings: () => false,
				acceptsLanguages: () => false,
				range: () => undefined,
				param: (name: string) =>
					name === 'agentId' ? agentMatch[1] : undefined,
				is: () => false,
				protocol: 'https',
				secure: true,
				ip: '127.0.0.1',
				ips: [],
				subdomains: [],
				hostname: url.hostname,
				fresh: false,
				stale: true,
				xhr: false,
				route: undefined,
				signedCookies: {},
				url: cleanPath + url.search,
				baseUrl: '',
				app: {} as any,
				res: {} as any,
				next: (() => {}) as any,
				query: Object.fromEntries(url.searchParams),
				cookies: {},
				secret: undefined,
			} as any;

			// Create a mock response object
			const mockRes = {
				status: (code: number) => ({json: (data: any) => data}),
				json: (data: any) => data,
				send: (data: any) => data,
				end: () => {},
				setHeader: () => {},
				getHeader: () => undefined,
				removeHeader: () => {},
				locals: {},
				append: () => {},
				attachment: () => {},
				cookie: () => {},
				clearCookie: () => {},
				download: () => {},
				format: () => {},
				get: () => undefined,
				header: () => {},
				links: () => {},
				location: () => {},
				redirect: () => {},
				render: () => {},
				sendFile: () => {},
				sendStatus: () => {},
				set: () => {},
				type: () => {},
				vary: () => {},
			} as any;

			const mockNext = () => {};

			try {
				await handleAgentRequest(
					mockReq,
					mockRes,
					mockNext,
					taskStore,
					agentExecutor,
					port,
					repositories,
				);
				return new Response(JSON.stringify({success: true}), {
					status: 200,
					headers,
				});
			} catch (error) {
				console.error('Error in agent request handler:', error);
				return new Response(
					JSON.stringify({
						error:
							error instanceof Error
								? error.message
								: 'Failed to handle agent request',
					}),
					{status: 500, headers},
				);
			}
		}

		return new Response('Not found', {status: 404, headers});
	} catch (error) {
		console.error('Error in Supabase Edge Function:', error);
		return new Response(
			JSON.stringify({
				error: error instanceof Error ? error.message : 'Internal server error',
			}),
			{status: 500, headers},
		);
	}
});

console.log('🚀 Timestep Server running with Custom Supabase Repositories');
console.log('📚 Available endpoints:');
console.log('  - GET /agents - List agents (using SupabaseAgentRepository)');
console.log('  - /agents/{agentId}/* - Dynamic agent A2A endpoints');
console.log('  - GET /chats - List chats (using SupabaseContextRepository)');
console.log('  - GET /health - Health check with repository info');
console.log(
	'  - GET /mcp_servers - List MCP servers (using SupabaseMcpServerRepository)',
);
console.log('  - GET /mcp_servers/{serverId} - MCP server health');
console.log(
	'  - GET /model_providers - List model providers (using SupabaseModelProviderRepository)',
);
console.log(
	'  - GET /model_providers/{providerId} - Get a specific model provider',
);
console.log(
	'  - GET /models - List models (via SupabaseModelProviderRepository)',
);
console.log('  - GET /tools - List tools (via SupabaseMcpServerRepository)');
console.log('  - GET /tools/{toolId} - Get specific tool information');
console.log('  - GET /traces - List traces (using default hardcoded data)');
console.log('  - GET /version - Timestep package version information');

/*
 * SQL Schema for Supabase Tables
 *
 * Run these commands in your Supabase SQL editor to create the required tables:
 */
export const supabaseSchemaSQL = `
-- Enable extension for UUIDs
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create agents table
CREATE TABLE agents (
  id UUID NOT NULL DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  name TEXT NOT NULL,
  instructions TEXT NOT NULL,
  handoff_description TEXT,
  handoff_ids JSONB,
  tool_ids JSONB NOT NULL,
  model TEXT NOT NULL,
  model_settings JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (user_id, id)
);

-- Create contexts table
CREATE TABLE contexts (
  id UUID NOT NULL DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  agent_id UUID NOT NULL,
  task_histories JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (user_id, id)
);

-- Create mcp_servers table
CREATE TABLE mcp_servers (
  id UUID NOT NULL DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  server_url TEXT,
  enabled BOOLEAN DEFAULT TRUE,
  auth_token TEXT,
  command TEXT,
  args JSONB,
  env JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (user_id, id)
);

-- Create model_providers table
CREATE TABLE model_providers (
  id UUID NOT NULL DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  provider TEXT NOT NULL,
  api_key TEXT,
  base_url TEXT NOT NULL,
  models_url TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (user_id, id)
);

-- Create indexes for better performance
CREATE INDEX idx_agents_user_id ON agents(user_id);
CREATE INDEX idx_agents_name ON agents(name);
CREATE INDEX idx_contexts_user_id ON contexts(user_id);
CREATE INDEX idx_contexts_agent_id ON contexts(agent_id);
CREATE INDEX idx_mcp_servers_user_id ON mcp_servers(user_id);
CREATE INDEX idx_mcp_servers_name ON mcp_servers(name);
CREATE INDEX idx_model_providers_user_id ON model_providers(user_id);
CREATE INDEX idx_model_providers_provider ON model_providers(provider);

-- Enable Row Level Security (optional)
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE contexts ENABLE ROW LEVEL SECURITY;
ALTER TABLE mcp_servers ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_providers ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated users (optional)
-- Basic per-user RLS: require matching user_id
CREATE POLICY "Users can access agents" ON agents FOR ALL TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can access contexts" ON contexts FOR ALL TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can access mcp_servers" ON mcp_servers FOR ALL TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can access model_providers" ON model_providers FOR ALL TO authenticated USING (user_id = auth.uid());
`;

/*
 * Environment Variables Required:
 *
 * Set these in your Supabase Edge Function environment:
 * - SUPABASE_URL: Your Supabase project URL
 * - SUPABASE_ANON_KEY: Your Supabase anon key
 * - PORT: Server port (optional, defaults to 3000)
 * - DENO_DEPLOYMENT_ID: Deployment identifier (auto-set by Supabase)
 * - DENO_REGION: Deployment region (auto-set by Supabase)
 */
