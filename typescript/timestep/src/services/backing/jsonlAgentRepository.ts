import {Repository} from './repository.js';
import {Agent} from '../../api/agentsApi.js';
import {JsonlRepository} from './jsonlRepository.js';
import {getTimestepPaths} from '../../utils.js';
import {getDefaultAgents} from '../../config/defaultAgents.js';
import * as fs from 'node:fs';
import * as path from 'node:path';

// Get timestep configuration paths
const timestepPaths = getTimestepPaths();

/**
 * JSONL file-based implementation of AgentRepository.
 * Stores agents as JSON objects in a .jsonl file, one agent per line.
 */
export class JsonlAgentRepository
	extends JsonlRepository<Agent, string>
	implements Repository<Agent, string>
{
	constructor() {
		super(timestepPaths.agentsConfig);
	}

	protected serialize(agent: Agent): string {
		return JSON.stringify(agent);
	}

	protected deserialize(line: string): Agent {
		return JSON.parse(line) as Agent;
	}

	protected getId(agent: Agent): string {
		return agent.id;
	}

	override async list(): Promise<Agent[]> {
		try {
			const agents = await super.list();
			if (agents.length > 0) {
				// console.log(`📋 Loaded ${agents.length} agents from ${this.filePath}`);
				return agents;
			}
		} catch (error) {
			console.warn(
				`Failed to read agents configuration from '${this.filePath}': ${error}. Using default configuration.`,
			);
		}

		// If no agents found or error reading, try to create default configuration
		// In restricted environments (like Supabase Edge Functions), this will fail gracefully
		const defaultAgents = getDefaultAgents();
		try {
			await this.createDefaultAgentsFile();
			console.log(
				`📋 Created default agents configuration with ${defaultAgents.length} agents`,
			);
		} catch (error) {
			console.warn(
				`Unable to create default configuration file (restricted environment): ${error}`,
			);
			console.log(
				`📋 Using in-memory default agents configuration with ${defaultAgents.length} agents`,
			);
		}

		return defaultAgents;
	}

	/**
	 * Create the agents configuration file with default agents
	 */
	private async createDefaultAgentsFile(): Promise<void> {
		try {
			// Ensure the directory exists
			const dir = path.dirname(this.filePath);
			if (!fs.existsSync(dir)) {
				fs.mkdirSync(dir, {recursive: true});
			}

			// Write the default agents as JSONL
			const defaultAgents = getDefaultAgents();
			const lines = defaultAgents.map(agent => this.serialize(agent));
			await this.writeLines(lines);
			console.log(
				`📋 Created default agents configuration at: ${this.filePath}`,
			);
		} catch (error) {
			console.warn(
				`Failed to create default agents configuration at '${this.filePath}': ${error}`,
			);
		}
	}
}
