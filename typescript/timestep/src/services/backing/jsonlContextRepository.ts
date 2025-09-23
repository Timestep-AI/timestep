import {Repository} from './repository.js';
import {Context} from '../../types/context.js';
import {JsonlRepository} from './jsonlRepository.js';
import {getTimestepPaths} from '../../utils.js';

// Get timestep configuration paths
const timestepPaths = getTimestepPaths();

/**
 * JSONL file-based implementation of ContextRepository.
 * Stores contexts as JSON objects in a .jsonl file, one context per line.
 */
export class JsonlContextRepository
	extends JsonlRepository<Context, string>
	implements Repository<Context, string>
{
	constructor() {
		super(timestepPaths.contexts);
	}

	protected serialize(context: Context): string {
		return JSON.stringify(context.toJSON());
	}

	protected deserialize(line: string): Context {
		const contextData = JSON.parse(line);
		return Context.fromJSON(contextData);
	}

	protected getId(context: Context): string {
		return context.contextId;
	}

	override async getOrCreate(
		contextId: string,
		agentId: string,
	): Promise<Context> {
		const existing = await this.load(contextId);
		if (existing) {
			return existing;
		}

		return new Context(contextId, agentId);
	}

	override async save(context: Context): Promise<void> {
		await super.save(context);
		// console.log(
		// 	`💾 Saved context ${context.contextId} with ${
		// 		Object.keys(context.taskHistories).length
		// 	} tasks`,
		// );
	}

	override async load(contextId: string): Promise<Context | null> {
		const context = await super.load(contextId);
		// if (context) {
		// 	console.log(
		// 		`🔍 Loaded context ${contextId} with ${
		// 			Object.keys(context.taskHistories || {}).length
		// 		} tasks`,
		// 	);
		// }
		return context;
	}
}
