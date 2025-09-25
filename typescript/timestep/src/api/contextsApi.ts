/**
 * Contexts API
 *
 * This module provides TypeScript interfaces and functions for managing
 * conversation contexts and chat sessions.
 *
 * Functions:
 * - listContexts() - List all contexts using the context service
 */

import {Context} from '../types/context.js';
import {ContextService} from '../services/contextService.js';
import {
	RepositoryContainer,
	DefaultRepositoryContainer,
} from '../services/backing/repositoryContainer.js';

/**
 * Response from the list contexts endpoint
 */
export interface ListContextsResponse {
	/** Array of context objects */
	data: Context[];
}

/**
 * List all contexts using the context service
 *
 * @param repositories Optional repository container for dependency injection. Defaults to DefaultRepositoryContainer
 * @returns Promise resolving to the list of contexts
 */
export async function listContexts(
	repositories: RepositoryContainer = new DefaultRepositoryContainer(),
): Promise<ListContextsResponse> {
	const contextService = new ContextService(repositories.contexts);

	try {
		const contexts = await contextService.listContexts();
		return {
			data: contexts,
		};
	} catch (error) {
		throw new Error(`Failed to read contexts: ${error}`);
	}
}

/**
 * Get a specific context by ID using the context service
 *
 * @param contextId The ID of the context to retrieve
 * @param repositories Optional repository container for dependency injection. Defaults to DefaultRepositoryContainer
 * @returns Promise resolving to the context if found, null otherwise
 */
export async function getContext(
	contextId: string,
	repositories: RepositoryContainer = new DefaultRepositoryContainer(),
): Promise<Context | null> {
	const contextService = new ContextService(repositories.contexts);

	try {
		return await contextService.getContext(contextId);
	} catch (error) {
		throw new Error(`Failed to get context: ${error}`);
	}
}
