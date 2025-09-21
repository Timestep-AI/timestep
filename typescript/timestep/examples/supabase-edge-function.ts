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
} from 'npm:@timestep-ai/timestep@2025.9.211411';

// Custom function to get agent card with correct Supabase base URL
async function getAgentCardForSupabase(
	agentId: string,
	baseUrl: string,
	repositories: any,
): Promise<any> {
	const {getAgent} = await import('npm:@timestep-ai/timestep@2025.9.211411');
	const agent = await getAgent(agentId, repositories);
	if (!agent) {
		throw new Error(`Agent ${agentId} not found`);
	}

	// Create agent card with correct base URL
	const agentCard = {
		capabilities: {streaming: true},
		defaultInputModes: ['text'],
		defaultOutputModes: ['text'],
		description: `A helpful AI agent powered by ${agent.name}`,
		name: agent.name,
		preferredTransport: 'JSONRPC',
		protocolVersion: '0.3.0',
		skills: [
			{
				id: 'hello_world',
				name: 'Returns hello world',
				description: 'just returns hello world',
				examples: ['hi', 'hello world'],
				tags: ['hello world'],
			},
		],
		supportsAuthenticatedExtendedCard: true,
		url: `${baseUrl}/agents/${agentId}/`,
		version: '1.0.0',
	};

	return agentCard;
}

/**
 * Supabase Agent Repository Implementation
 */
class SupabaseAgentRepository implements Repository<Agent, string> {
	constructor(private supabase: any, private userId: string | null) {}

	async list(): Promise<Agent[]> {
		if (!this.userId) return [];

		// First, check if user has any agents
		const {data: existingData, error: checkError} = await this.supabase
			.from('agents')
			.select('id')
			.eq('user_id', this.userId)
			.limit(1);

		if (checkError)
			throw new Error(`Failed to check existing agents: ${checkError.message}`);

		// Always ensure default agents exist and are up-to-date
		try {
			const {getDefaultAgents} = await import(
				'npm:@timestep-ai/timestep@2025.9.211411'
			);
			const defaultAgents = getDefaultAgents();

			// Get existing agents to compare
			const {data: existingAgents, error: listError} = await this.supabase
				.from('agents')
				.select('*')
				.eq('user_id', this.userId);

			if (listError) {
				console.warn(`Failed to list existing agents: ${listError.message}`);
			} else {
				// Update or create each default agent
				for (const defaultAgent of defaultAgents) {
					const existingAgent = existingAgents?.find(
						(a: any) => a.id === defaultAgent.id,
					);

					if (!existingAgent) {
						// Create new default agent
						await this.save(defaultAgent);
						console.log(`Created default agent: ${defaultAgent.name}`);
					} else {
						// Check if agent needs updating
						const toolIdsMatch =
							JSON.stringify(existingAgent.tool_ids || []) ===
							JSON.stringify((defaultAgent as any).toolIds || []);
						const handoffIdsMatch =
							JSON.stringify(existingAgent.handoff_ids || []) ===
							JSON.stringify((defaultAgent as any).handoffIds || []);
						const nameMatch = existingAgent.name === defaultAgent.name;
						const instructionsMatch =
							existingAgent.instructions === defaultAgent.instructions;

						const needsUpdate =
							!toolIdsMatch ||
							!handoffIdsMatch ||
							!nameMatch ||
							!instructionsMatch;

						console.log(
							`Agent ${defaultAgent.name}: toolIds match=${toolIdsMatch}, handoffIds match=${handoffIdsMatch}, name match=${nameMatch}, instructions match=${instructionsMatch}, needs update=${needsUpdate}`,
						);

						if (needsUpdate) {
							// Update existing agent with new defaults
							const updatedAgent = {
								...defaultAgent,
								id: existingAgent.id, // Keep the existing ID
							};
							await this.save(updatedAgent);
							console.log(`Updated agent: ${defaultAgent.name}`);
						}
					}
				}
			}
		} catch (saveError) {
			console.warn(`Failed to ensure default agents: ${saveError}`);
		}

		const {data, error} = await this.supabase
			.from('agents')
			.select('*')
			.eq('user_id', this.userId);
		if (error) throw new Error(`Failed to list agents: ${error.message}`);

		// Transform database field names to expected format
		return (data || []).map((agent: any) => ({
			...agent,
			handoffIds: agent.handoff_ids || [],
			toolIds: agent.tool_ids || [],
		}));
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

		if (!data) return null;

		// Transform database field names to expected format
		return {
			...data,
			handoffIds: data.handoff_ids || [],
			toolIds: data.tool_ids || [],
		};
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

		// First check if context exists
		const existing = await this.load(context.contextId);
		const now = new Date().toISOString();

		if (existing) {
			// Update existing context
			const {error} = await this.supabase
				.from('contexts')
				.update({
					task_histories: context.taskHistories,
					task_states: context.taskStates,
					tasks: context.tasks,
					updated_at: now,
				})
				.eq('user_id', this.userId)
				.eq('context_id', context.contextId);

			if (error) throw new Error(`Failed to update context: ${error.message}`);
		} else {
			// Insert new context
			const {error} = await this.supabase.from('contexts').insert({
				id: crypto.randomUUID(),
				user_id: this.userId,
				context_id: context.contextId,
				agent_id: context.agentId,
				task_histories: context.taskHistories,
				task_states: context.taskStates,
				tasks: context.tasks,
				created_at: now,
				updated_at: now,
			});

			if (error) throw new Error(`Failed to create context: ${error.message}`);
		}
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

		// First, check if user has any MCP servers
		const {data: existingData, error: checkError} = await this.supabase
			.from('mcp_servers')
			.select('id')
			.eq('user_id', this.userId)
			.limit(1);

		if (checkError)
			throw new Error(
				`Failed to check existing MCP servers: ${checkError.message}`,
			);

		console.log(
			`🔌 MCP server list called, existing servers: ${
				existingData?.length || 0
			}`,
		);

		// Always ensure the built-in MCP server exists, regardless of other servers
		const builtinServerId = '00000000-0000-0000-0000-000000000000';
		const hasBuiltinServer = existingData?.some(
			(server: any) => server.id === builtinServerId,
		);

		console.log(
			`🔌 Built-in MCP server exists: ${hasBuiltinServer}, baseUrl: ${this.baseUrl}`,
		);

		if (!hasBuiltinServer) {
			try {
				// For the built-in server, don't set a serverUrl - it's handled internally
				const builtinServer = {
					id: '00000000-0000-0000-0000-000000000000',
					name: 'Built-in MCP Server',
					description:
						'Built-in MCP server providing weather data, document tools, and thinking capabilities',
					serverUrl: '', // Empty for built-in server - handled internally
					enabled: true,
				};
				console.log(`🔌 Creating built-in MCP server:`, builtinServer);
				await this.save(builtinServer);
				console.log(`🔌 Created built-in MCP server in database`);
			} catch (e) {
				console.warn(`Failed to create built-in MCP server: ${e}`);
			}
		}

		// Create other defaults only if no MCP servers exist at all
		if (!existingData || existingData.length === 0) {
			try {
				const {getDefaultMcpServers} = await import(
					'npm:@timestep-ai/timestep@2025.9.211411'
				);
				const defaults = getDefaultMcpServers(this.baseUrl);
				// Skip the built-in server since we already handled it above
				const otherDefaults = defaults.filter(
					server => server.id !== builtinServerId,
				);
				for (const server of otherDefaults) {
					await this.save(server);
				}
				if (otherDefaults.length > 0) {
					console.log(
						`🔌 Created ${otherDefaults.length} additional default MCP servers in database`,
					);
				}
			} catch (e) {
				console.warn(`Failed to create default MCP servers: ${e}`);
			}
		}

		const {data, error} = await this.supabase
			.from('mcp_servers')
			.select('*')
			.eq('user_id', this.userId);
		if (error) throw new Error(`Failed to list MCP servers: ${error.message}`);

		// Decrypt auth tokens if needed
		const {isEncryptedSecret, decryptSecret} = await import(
			'npm:@timestep-ai/timestep@2025.9.211411'
		);

		const servers = await Promise.all(
			(data || []).map(async (row: any) => {
				let authToken = row.auth_token;
				if (authToken && isEncryptedSecret(authToken)) {
					try {
						authToken = await decryptSecret(authToken);
					} catch (e) {
						console.warn(
							`Failed to decrypt auth token for MCP server ${row.name}: ${e}`,
						);
						authToken = null;
					}
				}
				// For built-in server, don't set serverUrl - it's handled internally
				const isBuiltin = row.id === '00000000-0000-0000-0000-000000000000';
				return {
					id: row.id,
					name: row.name,
					description: row.description ?? row.name,
					serverUrl: isBuiltin ? '' : row.server_url ?? '',
					enabled: row.enabled ?? true,
					authToken: authToken,
				};
			}),
		);

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

		// Decrypt auth token if needed
		const {isEncryptedSecret, decryptSecret} = await import(
			'npm:@timestep-ai/timestep@2025.9.211411'
		);

		let authToken = data.auth_token;
		if (authToken && isEncryptedSecret(authToken)) {
			try {
				authToken = await decryptSecret(authToken);
			} catch (e) {
				console.warn(
					`Failed to decrypt auth token for MCP server ${data.name}: ${e}`,
				);
				authToken = null;
			}
		}

		// Apply the same transformation logic as the list method
		// For built-in server, don't set serverUrl - it's handled internally
		const isBuiltin = data.id === '00000000-0000-0000-0000-000000000000';
		return {
			id: data.id,
			name: data.name,
			description: data.description ?? data.name,
			serverUrl: isBuiltin ? '' : data.server_url ?? '',
			enabled: data.enabled ?? true,
			authToken: authToken,
		};
	}

	async save(server: McpServer): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		// Persist server using snake_case column mapping
		const toSave: any = {
			id: server.id,
			user_id: this.userId,
			name: server.name,
			description: (server as any).description ?? server.name,
			server_url: (server as any).server_url ?? (server as any).serverUrl,
			enabled: server.enabled,
		};
		const {isEncryptedSecret, encryptSecret} = await import(
			'npm:@timestep-ai/timestep@2025.9.211411'
		);

		// Handle auth token - encrypt if provided, set to null if not
		if (
			(server as any).auth_token !== undefined ||
			(server as any).authToken !== undefined
		) {
			let token = ((server as any).auth_token ?? (server as any).authToken) as
				| string
				| undefined;
			if (token && !isEncryptedSecret(token)) {
				try {
					token = await encryptSecret(token);
				} catch {}
			}
			toSave.auth_token = token ?? null;
		} else {
			toSave.auth_token = null;
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

		// First, check if user has any model providers
		const {data: existingData, error: checkError} = await this.supabase
			.from('model_providers')
			.select('id')
			.eq('user_id', this.userId)
			.limit(1);

		if (checkError)
			throw new Error(
				`Failed to check existing model providers: ${checkError.message}`,
			);

		// Only create defaults if no model providers exist
		if (!existingData || existingData.length === 0) {
			try {
				const {getDefaultModelProviders} = await import(
					'npm:@timestep-ai/timestep@2025.9.211411'
				);
				const defaults = getDefaultModelProviders();
				for (const p of defaults) {
					await this.save(p);
				}
			} catch (e) {
				console.warn(`Failed to create default model providers: ${e}`);
			}
		}

		const {data, error} = await this.supabase
			.from('model_providers')
			.select('*')
			.eq('user_id', this.userId);

		if (error)
			throw new Error(`Failed to list model providers: ${error.message}`);

		// Decrypt API keys if needed
		const {isEncryptedSecret, decryptSecret} = await import(
			'npm:@timestep-ai/timestep@2025.9.211411'
		);

		const providers = await Promise.all(
			(data || []).map(async (row: any) => {
				let apiKey = row.api_key;
				if (apiKey && isEncryptedSecret(apiKey)) {
					try {
						apiKey = await decryptSecret(apiKey);
					} catch (e) {
						console.warn(
							`Failed to decrypt API key for provider ${row.provider}: ${e}`,
						);
						apiKey = null;
					}
				}
				return {
					id: row.id,
					provider: row.provider,
					api_key: apiKey,
					base_url: row.base_url,
					models_url: row.models_url,
				};
			}),
		);

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

		if (!data) return null;

		// Decrypt API key if needed
		const {isEncryptedSecret, decryptSecret} = await import(
			'npm:@timestep-ai/timestep@2025.9.211411'
		);

		let apiKey = data.api_key;
		if (apiKey && isEncryptedSecret(apiKey)) {
			try {
				apiKey = await decryptSecret(apiKey);
			} catch (e) {
				console.warn(
					`Failed to decrypt API key for provider ${data.provider}: ${e}`,
				);
				apiKey = null;
			}
		}

		// Return decrypted data
		return {
			...data,
			api_key: apiKey,
		};
	}

	async save(provider: ModelProvider): Promise<void> {
		if (!this.userId) throw new Error('Unauthenticated: user_id required');
		// Map to snake_case; encrypt apiKey if provided
		const toSave: any = {
			id: provider.id,
			user_id: this.userId,
			provider: provider.provider,
			base_url: (provider as any).base_url ?? (provider as any).baseUrl,
			models_url: (provider as any).models_url ?? (provider as any).modelsUrl,
		};
		const {isEncryptedSecret, encryptSecret} = await import(
			'npm:@timestep-ai/timestep@2025.9.211411'
		);
		if (
			(provider as any).api_key !== undefined ||
			(provider as any).apiKey !== undefined
		) {
			let key = ((provider as any).api_key ?? (provider as any).apiKey) as
				| string
				| undefined;
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
	constructor(private supabase: any, public userId: string | null) {}

	async load(taskId: string): Promise<any | undefined> {
		console.log(`📋 SupabaseTaskStore.load(${taskId})`);

		if (!this.userId) {
			console.log(
				`📋 SupabaseTaskStore.load(${taskId}) -> NOT FOUND (no user)`,
			);
			return undefined;
		}

		const {data, error} = await this.supabase
			.from('tasks')
			.select('*')
			.eq('task_id', taskId)
			.eq('user_id', this.userId)
			.single();

		if (error && error.code !== 'PGRST116') {
			console.error(`📋 SupabaseTaskStore.load(${taskId}) -> ERROR:`, error);
			return undefined;
		}

		if (!data) {
			console.log(`📋 SupabaseTaskStore.load(${taskId}) -> NOT FOUND`);
			return undefined;
		}

		console.log(`📋 SupabaseTaskStore.load(${taskId}) -> FOUND`);
		return data.task_data;
	}

	async save(task: any): Promise<void> {
		console.log(`📋 SupabaseTaskStore.save(${task.id})`);

		if (!this.userId) {
			console.log(`📋 SupabaseTaskStore.save(${task.id}) -> SKIPPED (no user)`);
			return;
		}

		const {error} = await this.supabase.from('tasks').upsert(
			[
				{
					task_id: task.id,
					user_id: this.userId,
					task_data: task,
					created_at: new Date().toISOString(),
					updated_at: new Date().toISOString(),
				},
			],
			{onConflict: 'task_id,user_id'},
		);

		if (error) {
			console.error(`📋 SupabaseTaskStore.save(${task.id}) -> ERROR:`, error);
		} else {
			console.log(`📋 SupabaseTaskStore.save(${task.id}) -> SAVED`);
		}
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
const taskStore = new SupabaseTaskStore(supabase, null); // Will be set per request

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

	// Respect original protocol behind the proxy (prefer https on Supabase)
	const forwardedProto =
		request.headers.get('x-forwarded-proto') ||
		request.headers.get('X-Forwarded-Proto');
	const inferredProto =
		forwardedProto ||
		(url.hostname.endsWith('.supabase.co')
			? 'https'
			: url.protocol.replace(':', ''));
	const scheme = inferredProto.endsWith(':')
		? inferredProto
		: `${inferredProto}:`;

	// Generate base URL for MCP servers from the current request
	const baseUrl = `${scheme}//${url.host}/${functionName}`;

	// Generate agent base URL for agent cards (without the /functions/v1/ prefix)
	const agentBaseUrl = `${scheme}//${url.host}/${functionName}`;

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
			const result = await listAgents(repositories as any);
			return new Response(JSON.stringify(result.data), {status: 200, headers});
		}

		if (cleanPath === '/chats') {
			const result = await listContexts(repositories as any);
			return new Response(JSON.stringify(result.data), {status: 200, headers});
		}

		if (cleanPath === '/mcp_servers') {
			const result = await listMcpServers(repositories as any);
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
			const result = await listModelProviders(repositories as any);
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
					const provider = await getModelProvider(
						providerId,
						repositories as any,
					);
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
			const result = await listModels(repositories as any);
			return new Response(JSON.stringify(result.data), {status: 200, headers});
		}

		if (cleanPath === '/tools') {
			const result = await listTools(repositories as any);
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
					'npm:@timestep-ai/timestep@2025.9.211411'
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
					repositories as any,
				);

				if ('error' in listResponse && listResponse.error) {
					return new Response(
						JSON.stringify({
							error: `Failed to list tools from server ${serverId}`,
							details: listResponse.error,
						}),
						{status: 500, headers},
					);
				}

				// Find the specific tool
				const tools =
					('result' in listResponse && listResponse.result?.tools) || [];
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
					'npm:@timestep-ai/timestep@2025.9.211411'
				);

				const result = await handleMcpServerRequest(
					serverId,
					{
						jsonrpc: '2.0',
						method: 'tools/call',
						params: {name: toolName, arguments: args},
						id,
					},
					repositories as any,
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
					'npm:@timestep-ai/timestep@2025.9.211411'
				);

				if (request.method === 'POST') {
					const body = await request.json().catch(() => ({}));
					const result = await handleMcpServerRequest(
						serverId,
						body,
						repositories as any,
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
					'npm:@timestep-ai/timestep@2025.9.211411'
				);
				const server = await getMcpServer(serverId, repositories as any);

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

		// Handle dynamic agent routes - proxy to A2A Express app like server.ts
		console.log(`🔍 Checking if path matches agent route: ${cleanPath}`);
		// Check for both /agents/... and /server/agents/... patterns (Supabase function URL structure)
		const agentMatch = cleanPath.match(
			/^\/(?:server\/)?agents\/([^\/]+)(?:\/.*)?$/,
		);
		if (agentMatch) {
			console.log(
				`🔍 Agent route matched! agentId: ${agentMatch[1]}, full path: ${cleanPath}`,
			);
			const agentId = agentMatch[1];

			// Compute sub-path for the agent app (strip /agents/{agentId} prefix, accounting for optional /server prefix)
			const agentPrefix = `/agents/${agentId}`;
			const serverAgentPrefix = `/server${agentPrefix}`;
			let agentSubPath: string;

			if (cleanPath.startsWith(serverAgentPrefix)) {
				// Path has /server/agents/{agentId} prefix
				agentSubPath = cleanPath.slice(serverAgentPrefix.length) || '/';
			} else if (cleanPath.startsWith(agentPrefix)) {
				// Path has /agents/{agentId} prefix
				agentSubPath = cleanPath.slice(agentPrefix.length) || '/';
			} else {
				// Fallback
				agentSubPath = cleanPath;
			}
			console.log(
				`🔍 Agent prefixes checked: ${agentPrefix}, ${serverAgentPrefix}, subPath: ${agentSubPath}`,
			);

			// Create a mock Express-style request object that satisfies the Request interface
			// Get the request body as text for compatibility
			let requestBodyText = '';
			let requestBody: Uint8Array | null = null;
			if (request.method !== 'GET') {
				try {
					requestBodyText = await request.text();
					requestBody = new TextEncoder().encode(requestBodyText);
				} catch (e) {
					requestBodyText = '';
					requestBody = new Uint8Array(0);
				}
			}

			const mockReq = {
				method: request.method,
				path: agentSubPath,
				originalUrl: agentSubPath + url.search,
				params: {agentId: agentId},
				body:
					request.method !== 'GET' ? JSON.parse(requestBodyText || '{}') : {},
				headers: Object.fromEntries(Array.from(request.headers.entries())),
				// Add required Express Request methods as stubs
				get: (name: string) => request.headers.get(name),
				header: (name: string) => request.headers.get(name),
				accepts: () => false,
				acceptsCharsets: () => false,
				acceptsEncodings: () => false,
				acceptsLanguages: () => false,
				range: () => undefined,
				param: (name: string) => (name === 'agentId' ? agentId : undefined),
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
				url: agentSubPath + url.search,
				baseUrl: '/agents',
				app: {} as any,
				res: {} as any,
				next: (() => {}) as any,
				query: Object.fromEntries(url.searchParams),
				cookies: {},
				secret: undefined,
				// Add stream properties for Express compatibility
				readable: true,
				readableHighWaterMark: 16384,
				readableLength: requestBody ? requestBody.length : 0,
				_buffer: requestBody,
				destroyed: false,
				// Mock pipe method
				pipe: (destination: any) => {
					if (requestBody) {
						destination.write(requestBody);
					}
					destination.end();
					return destination;
				},
				// Mock on method for event listeners
				on: (event: string, listener: Function) => {
					// No-op for basic compatibility
					return mockReq;
				},
				// Add pipes property that was causing the error
				pipes: [],
				// Add other stream properties
				readableFlowing: null,
				readableEnded: false,
				readablePaused: false,
			} as any;

			// Create a proper response handler that captures the response
			let responseData: any = null;
			let responseStatus = 200;
			let responseHeaders: Record<string, string> = {...headers};
			let isStreaming = false;
			let responseEnded = false;
			let stream: ReadableStream<Uint8Array> | null = null;
			let streamController: ReadableStreamDefaultController<Uint8Array> | null =
				null;
			let responseResolved = false;
			let responseResolve:
				| ((r: {
						type: 'stream' | 'body';
						status: number;
						headers: Record<string, string>;
				  }) => void)
				| null = null;
			const responsePromise = new Promise<{
				type: 'stream' | 'body';
				status: number;
				headers: Record<string, string>;
			}>(resolve => {
				responseResolve = r => {
					if (!responseResolved) {
						responseResolved = true;
						resolve(r);
					}
				};
			});

			const ensureStream = () => {
				if (!stream) {
					stream = new ReadableStream<Uint8Array>({
						start(controller) {
							streamController = controller;
						},
					});
				}
			};

			const mockRes = {
				status: (code: number) => {
					console.log(`🔍 MockRes.status called with code: ${code}`);
					responseStatus = code;
					return mockRes; // Return self for chaining
				},
				json: (data: any) => {
					console.log(`🔍 MockRes.json called with data:`, data);
					responseData = data;
					responseEnded = true;
					return mockRes;
				},
				send: (data: any) => {
					console.log(`🔍 MockRes.send called with data:`, data);
					responseData = data;
					responseEnded = true;
					return mockRes;
				},
				end: (data?: any) => {
					console.log(`🔍 MockRes.end called with data:`, data);
					if (data !== undefined) {
						responseData = data;
					}
					responseEnded = true;
					if (streamController) {
						try {
							streamController.close();
						} catch {}
					}
					return mockRes;
				},
				setHeader: (name: string, value: string) => {
					console.log(`🔍 MockRes.setHeader called: ${name} = ${value}`);
					responseHeaders[name] = value;
					// Detect streaming responses
					if (
						name.toLowerCase() === 'content-type' &&
						value.includes('text/event-stream')
					) {
						isStreaming = true;
						ensureStream();
					}
					return mockRes;
				},
				getHeader: (name: string) => responseHeaders[name],
				removeHeader: (name: string) => {
					delete responseHeaders[name];
					return mockRes;
				},
				locals: {},
				append: () => mockRes,
				attachment: () => mockRes,
				cookie: () => mockRes,
				clearCookie: () => mockRes,
				download: () => mockRes,
				format: () => mockRes,
				get: () => undefined,
				header: (name?: any, value?: any) => {
					if (typeof name === 'string' && typeof value === 'string') {
						return (mockRes as any).setHeader(name, value);
					}
					if (name && typeof name === 'object') {
						for (const [k, v] of Object.entries(name)) {
							(mockRes as any).setHeader(k, String(v));
						}
					}
					return mockRes;
				},
				links: () => mockRes,
				location: () => mockRes,
				redirect: () => mockRes,
				render: () => mockRes,
				sendFile: () => mockRes,
				sendStatus: () => mockRes,
				set: (name: any, value?: any) => {
					if (typeof name === 'string' && typeof value === 'string') {
						return (mockRes as any).setHeader(name, value);
					}
					if (name && typeof name === 'object') {
						for (const [k, v] of Object.entries(name)) {
							(mockRes as any).setHeader(k, String(v));
						}
					}
					return mockRes;
				},
				type: (mime: string) => {
					if (mime) {
						(mockRes as any).setHeader('Content-Type', mime);
					}
					return mockRes;
				},
				vary: () => mockRes,
				// Add missing Express response methods that A2A might need
				flushHeaders: () => {
					console.log('🔍 MockRes.flushHeaders called');
					return mockRes;
				},
				flush: () => {
					console.log('🔍 MockRes.flush called');
					return mockRes;
				},
				on: (_event: string, _handler: (...args: any[]) => void) => {
					// No-op bridge for event listeners
					return mockRes;
				},
				write: (data: any) => {
					console.log(`🔍 MockRes.write called with data:`, data);
					if (responseData === null) {
						responseData = '';
					}
					const enc = new TextEncoder();
					const chunkUint8 =
						data instanceof Uint8Array
							? data
							: enc.encode(typeof data === 'string' ? data : String(data));
					responseData += new TextDecoder().decode(chunkUint8);
					if (streamController) {
						streamController.enqueue(chunkUint8);
					}
					return true;
				},
				writeHead: (
					statusCode: number,
					statusMessage?: string,
					headers?: any,
				) => {
					console.log(
						`🔍 MockRes.writeHead called with status: ${statusCode}, message: ${statusMessage}, headers:`,
						headers,
					);
					responseStatus = statusCode;
					if (headers) {
						Object.assign(responseHeaders, headers);
						const ct = Object.entries(headers).find(
							([k]) => k.toLowerCase() === 'content-type',
						);
						if (
							ct &&
							typeof ct[1] === 'string' &&
							ct[1].includes('text/event-stream')
						) {
							isStreaming = true;
							ensureStream();
							if (responseResolve)
								responseResolve({
									type: 'stream',
									status: responseStatus,
									headers: responseHeaders,
								});
						}
					}
					return mockRes;
				},
				finished: false,
				writable: true,
				writableEnded: false,
				writableFinished: false,
				writableHighWaterMark: 16384,
				writableLength: 0,
				writableObjectMode: false,
				writableCorked: 0,
				destroyed: false,
				readable: false,
				readableEnded: false,
				readableFlowing: null,
				readableHighWaterMark: 16384,
				readableLength: 0,
				readableObjectMode: false,
				readableAborted: false,
				readableDidRead: false,
				readableEncoding: null,
				readablePaused: false,
				readablePipes: [],
				readablePipesCount: 0,
				readableReadable: false,
				readableResumeScheduled: false,
				readableDestroyed: false,
				// Add a catch-all method to see if any other methods are being called
				[Symbol.iterator]: function* () {
					yield* [];
				},
			} as any;

			const mockNext = (err?: any) => {
				console.log(`🔍 MockNext called with error:`, err);
				if (err) {
					console.error(`🔍 MockNext error:`, err);
				}
			};

			try {
				// Check if agent exists first
				const timestepModule = await import(
					'npm:@timestep-ai/timestep@2025.9.211411'
				);
				const isAgentAvailable = timestepModule.isAgentAvailable;
				if (!(await isAgentAvailable(agentId, repositories as any))) {
					console.log(`❌ Agent ${agentId} not found`);
					return new Response(
						JSON.stringify({
							error: 'Agent not found',
							agentId: agentId,
						}),
						{status: 404, headers},
					);
				}

				// Extract port from the base URL for agent card generation
				const urlObj = new URL(agentBaseUrl);
				const port = urlObj.port
					? parseInt(urlObj.port)
					: urlObj.protocol === 'https:'
					? 443
					: 80;

				// For Supabase, the agent base URL should point to the function URL
				const supabaseAgentBaseUrl = agentBaseUrl.replace(
					'/server',
					'/functions/v1/server',
				);

				// Use handleAgentMessage directly for messaging

				// Check if this is an agent card request - handle it separately
				if (cleanPath.endsWith('/.well-known/agent-card.json')) {
					console.log(`🔍 Handling agent card request directly`);

					// Get the agent card directly
					const agentCard = await getAgentCardForSupabase(
						agentId,
						supabaseAgentBaseUrl,
						repositories as any,
					);

					console.log(`🔍 Agent card generated:`, agentCard);

					// Set response data directly
					responseData = agentCard;
					responseStatus = 200;
					responseHeaders['Content-Type'] = 'application/json';
					responseEnded = true;
				} else {
					console.log(`🔍 Handling A2A request: ${cleanPath}`);
					// Handle JSON-RPC message/stream request
					console.log(`🔍 Processing JSON-RPC request for agent ${agentId}`);
					console.log(`🔍 Request body:`, mockReq.body);

					const jsonRpcRequest = mockReq.body;
					if (jsonRpcRequest.method === 'message/stream') {
						// Use the actual A2A SDK streaming logic
						try {
							// Debug: Log agent details before creating request handler
							console.log(`🔍 Creating request handler for agent: ${agentId}`);
							console.log(
								`🔍 Available agents:`,
								(await repositories.agents.list()).map(a => ({
									id: a.id,
									name: a.name,
									toolIds: a.toolIds,
									handoffIds: a.handoffIds,
								})),
							);
							console.log(
								`🔍 Available MCP servers:`,
								(await repositories.mcpServers.list()).map(s => ({
									id: s.id,
									name: s.name,
									enabled: s.enabled,
									serverUrl: s.serverUrl,
								})),
							);

							// Set userId for task store
							taskStore.userId = userId;

							// Create the request handler for this agent
							const timestepModule = await import(
								'npm:@timestep-ai/timestep@2025.9.211411'
							);
							const createAgentRequestHandler =
								timestepModule.createAgentRequestHandler;
							const requestHandler = await createAgentRequestHandler(
								agentId,
								taskStore,
								agentExecutor,
								port,
								repositories as any,
							);

							// Extract message params from JSON-RPC request
							const messageParams = jsonRpcRequest.params;

							// Set up SSE response headers
							responseHeaders['Content-Type'] = 'text/event-stream';
							responseHeaders['Cache-Control'] = 'no-cache';
							responseHeaders['Connection'] = 'keep-alive';
							responseHeaders['Access-Control-Allow-Origin'] = '*';
							responseHeaders['Access-Control-Allow-Methods'] =
								'GET, POST, PUT, DELETE, OPTIONS';
							responseHeaders['Access-Control-Allow-Headers'] =
								'authorization, x-client-info, apikey, content-type';

							// Start the streaming response
							responseStatus = 200;
							isStreaming = true;

							// Create a ReadableStream that yields the actual stream events
							stream = new ReadableStream<Uint8Array>({
								async start(controller) {
									try {
										console.log(
											`🔍 Starting message stream for agent ${agentId}`,
										);

										// Call sendMessageStream and iterate over the events
										for await (const event of requestHandler.sendMessageStream(
											messageParams,
										)) {
											console.log(`🔍 Stream event:`, event);

											// Format as JSON-RPC SSE response
											const jsonRpcResponse = {
												jsonrpc: '2.0',
												id: jsonRpcRequest.id,
												result: event,
											};
											const sseData = `data: ${JSON.stringify(
												jsonRpcResponse,
											)}\n\n`;
											const encoder = new TextEncoder();
											controller.enqueue(encoder.encode(sseData));
										}

										console.log(`🔍 Stream completed for agent ${agentId}`);
										controller.close();
									} catch (error) {
										console.error(
											`🔍 Error in stream for agent ${agentId}:`,
											error,
										);
										const errorResponse = {
											jsonrpc: '2.0',
											id: jsonRpcRequest.id,
											error: {
												code: -32603,
												message: 'Internal error',
												data:
													error instanceof Error
														? error.message
														: 'Unknown error',
											},
										};
										const sseData = `data: ${JSON.stringify(
											errorResponse,
										)}\n\n`;
										const encoder = new TextEncoder();
										controller.enqueue(encoder.encode(sseData));
										controller.close();
									}
								},
							});

							responseEnded = true;
						} catch (error) {
							console.error(
								`🔍 Error setting up stream for agent ${agentId}:`,
								error,
							);
							const errorResponse = {
								jsonrpc: '2.0',
								id: jsonRpcRequest.id,
								error: {
									code: -32603,
									message: 'Internal error',
									data:
										error instanceof Error ? error.message : 'Unknown error',
								},
							};
							responseData = errorResponse;
							responseStatus = 500;
							responseHeaders['Content-Type'] = 'application/json';
							responseEnded = true;
						}
					} else {
						// Unknown method
						const errorResponse = {
							jsonrpc: '2.0',
							id: jsonRpcRequest.id,
							error: {
								code: -32601,
								message: 'Method not found',
							},
						};
						responseData = errorResponse;
						responseStatus = 200;
						responseHeaders['Content-Type'] = 'application/json';
						responseEnded = true;
					}
				}

				console.log(`🔍 Response ended: ${responseEnded}, data:`, responseData);
				console.log(
					`🔍 Response status: ${responseStatus}, headers:`,
					responseHeaders,
				);
				console.log(`🔍 Is streaming: ${isStreaming}`);

				// If streaming, return a streaming Response and bypass default JSON header
				if (isStreaming && stream) {
					const sseHeaders: Record<string, string> = {
						'Content-Type': 'text/event-stream',
						'Cache-Control': 'no-cache',
						Connection: 'keep-alive',
						...Object.fromEntries(
							Object.entries(responseHeaders).filter(
								([k]) => k.toLowerCase() !== 'content-type',
							),
						),
					};
					console.log('🔍 Returning SSE stream with headers:', sseHeaders);
					// Mark as ended for non-stream return paths
					responseEnded = true;
					return new Response(stream as any, {
						status: responseStatus,
						headers: sseHeaders,
					});
				}

				// If no response data was captured, fail fast - no fallbacks
				if (!responseData && !responseEnded) {
					console.error(
						`❌ No response data captured from handleAgentRequest for agent ${agentId}`,
					);
					return new Response(
						JSON.stringify({
							error: 'Agent request failed',
							message: 'No response data captured from agent handler',
							agentId: agentId,
						}),
						{status: 500, headers},
					);
				}

				// Return the response from handleAgentRequest
				// Avoid forcing JSON header if upstream set something else
				const finalHeaders = {
					...responseHeaders,
				};
				// If upstream intended streaming but didn't call writeHead earlier, infer from headers
				const ctHeader = Object.entries(finalHeaders).find(
					([k]) => k.toLowerCase() === 'content-type',
				);
				if (
					!isStreaming &&
					ctHeader &&
					typeof ctHeader[1] === 'string' &&
					ctHeader[1].includes('text/event-stream') &&
					stream
				) {
					console.log(
						'🔍 Upstream indicated SSE via headers; returning stream',
					);
					return new Response(stream as any, {
						status: responseStatus,
						headers: finalHeaders,
					});
				}
				return new Response(
					typeof responseData === 'string'
						? responseData
						: JSON.stringify(responseData),
					{
						status: responseStatus,
						headers: finalHeaders,
					},
				);
			} catch (error) {
				console.error(`Error in agent request handler for ${agentId}:`, error);
				return new Response(
					JSON.stringify({
						error: 'Agent request failed',
						message: error instanceof Error ? error.message : 'Unknown error',
						agentId: agentId,
					}),
					{status: 500, headers},
				);
			}
		}

		console.log(
			`🔍 No route matched for path: ${cleanPath}, method: ${request.method}`,
		);
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
  handoff_ids JSONB DEFAULT '[]',
  tool_ids JSONB NOT NULL DEFAULT '[]',
  model TEXT NOT NULL,
  model_settings JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (user_id, id)
);

-- Create contexts table
CREATE TABLE contexts (
  id UUID NOT NULL DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  context_id TEXT NOT NULL,
  agent_id UUID NOT NULL,
  task_histories JSONB DEFAULT '{}',
  task_states JSONB DEFAULT '{}',
  tasks JSONB DEFAULT '[]',
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

-- Create tasks table
CREATE TABLE tasks (
  id UUID NOT NULL DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  task_id TEXT NOT NULL,
  task_data JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (user_id, task_id)
);

-- Create indexes for better performance
CREATE INDEX idx_agents_user_id ON agents(user_id);
CREATE INDEX idx_agents_name ON agents(name);
CREATE INDEX idx_contexts_user_id ON contexts(user_id);
CREATE INDEX idx_contexts_context_id ON contexts(context_id);
CREATE INDEX idx_contexts_agent_id ON contexts(agent_id);
CREATE INDEX idx_mcp_servers_user_id ON mcp_servers(user_id);
CREATE INDEX idx_mcp_servers_name ON mcp_servers(name);
CREATE INDEX idx_model_providers_user_id ON model_providers(user_id);
CREATE INDEX idx_model_providers_provider ON model_providers(provider);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_task_id ON tasks(task_id);

-- Enable Row Level Security (optional)
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE contexts ENABLE ROW LEVEL SECURITY;
ALTER TABLE mcp_servers ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at columns
CREATE TRIGGER update_agents_updated_at 
    BEFORE UPDATE ON agents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_contexts_updated_at 
    BEFORE UPDATE ON contexts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mcp_servers_updated_at 
    BEFORE UPDATE ON mcp_servers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_providers_updated_at 
    BEFORE UPDATE ON model_providers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at 
    BEFORE UPDATE ON tasks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create policies for authenticated users (optional)
-- Basic per-user RLS: require matching user_id
CREATE POLICY "Users can access agents" ON agents FOR ALL TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can access contexts" ON contexts FOR ALL TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can access mcp_servers" ON mcp_servers FOR ALL TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can access model_providers" ON model_providers FOR ALL TO authenticated USING (user_id = auth.uid());
CREATE POLICY "Users can access tasks" ON tasks FOR ALL TO authenticated USING (user_id = auth.uid());
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
