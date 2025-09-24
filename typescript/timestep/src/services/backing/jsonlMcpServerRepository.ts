import {Repository} from './repository.js';
import {McpServer} from '../../api/mcpServersApi.js';
import {JsonlRepository} from './jsonlRepository.js';
import {getTimestepPaths} from '../../utils.js';
import {getDefaultMcpServers} from '../../config/defaultMcpServers.js';
import * as fs from 'node:fs';
import * as path from 'node:path';

// Get timestep configuration paths
const timestepPaths = getTimestepPaths();

/**
 * JSONL file-based implementation of McpServerRepository.
 * Stores MCP servers as JSON objects in a .jsonl file, one server per line.
 */
export class JsonlMcpServerRepository
	extends JsonlRepository<McpServer, string>
	implements Repository<McpServer, string>
{
	private baseUrl?: string;

	constructor(baseUrl?: string) {
		super(timestepPaths.mcpServers);
		this.baseUrl = baseUrl;
	}

	protected serialize(server: McpServer): string {
		return JSON.stringify(server);
	}

	protected deserialize(line: string): McpServer {
		return JSON.parse(line) as McpServer;
	}

	protected getId(server: McpServer): string {
		return server.id;
	}

	override async save(server: McpServer): Promise<void> {
		const toSave: McpServer = {...server};
		if (toSave.authToken) {
			const {isEncryptedSecret, encryptSecret} = await import('../../utils.js');
			if (!isEncryptedSecret(toSave.authToken)) {
				try {
					toSave.authToken = await encryptSecret(toSave.authToken);
				} catch (error) {
					console.warn('Failed to encrypt MCP server auth token:', error);
				}
			}
		}
		return super.save(toSave);
	}

	override async list(): Promise<McpServer[]> {
		try {
			const servers = await super.list();
			if (servers.length > 0) {
				return servers;
			}
		} catch (error) {
			console.warn(
				`Failed to read MCP servers configuration from '${this.filePath}': ${error}. Using default configuration.`,
			);
		}

		// If no servers found or error reading, try to create default configuration
		// In restricted environments (like Supabase Edge Functions), this will fail gracefully
		const defaultServers = getDefaultMcpServers(this.baseUrl);
		try {
			await this.createDefaultMcpServersFile(defaultServers);
		} catch (error) {
			console.warn(
				`Unable to create default configuration file (restricted environment): ${error}`,
			);
		}

		return defaultServers;
	}

	/**
	 * Create the MCP servers configuration file with default servers
	 */
	private async createDefaultMcpServersFile(
		servers: McpServer[],
	): Promise<void> {
		try {
			// Ensure the directory exists
			const dir = path.dirname(this.filePath);
			if (!fs.existsSync(dir)) {
				fs.mkdirSync(dir, {recursive: true});
			}

			// Write the default MCP servers as JSONL
			const lines = servers.map(server => this.serialize(server));
			await this.writeLines(lines);
		} catch (error) {
			console.warn(
				`Failed to create default MCP servers configuration at '${this.filePath}': ${error}`,
			);
		}
	}
}
