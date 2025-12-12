/** Common tools for agents, including web search using Firecrawl. */

import { z } from 'zod';
import Firecrawl from '@mendable/firecrawl-js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { CallToolRequest, CallToolResultSchema } from '@modelcontextprotocol/sdk/types.js';
import type { ChatCompletionMessageToolCall } from 'openai/resources/chat/completions';
import { handoff as handoffTool } from '../a2a/handoff.js';

interface SearchResult {
  title?: string;
  url?: string;
  description?: string;
}

interface SearchResponse {
  success?: boolean;
  // Handle both response structures:
  // 1. { data: SearchResult[] } - direct array
  // 2. { data: { web: SearchResult[] } } - nested structure (matches Python SDK)
  data?: SearchResult[] | { web?: SearchResult[] };
  warning?: string;
  error?: string;
}

function getFirecrawlClient(): Firecrawl {
  const apiKey = process.env.FIRECRAWL_API_KEY;
  if (!apiKey) {
    throw new Error(
      'FIRECRAWL_API_KEY environment variable is required for web search. ' +
      'Please set it to your Firecrawl API key.'
    );
  }
  return new Firecrawl({ apiKey });
}

function mapSearchContextSizeToLimit(searchContextSize: 'low' | 'medium' | 'high'): number {
  const mapping: Record<'low' | 'medium' | 'high', number> = {
    low: 5,
    medium: 10,
    high: 20,
  };
  return mapping[searchContextSize] || 10;
}

function extractDomain(url: string): string {
  try {
    const urlObj = new URL(url);
    let domain = urlObj.hostname.toLowerCase();
    // Remove www. prefix for comparison
    if (domain.startsWith('www.')) {
      domain = domain.substring(4);
    }
    return domain;
  } catch {
    return '';
  }
}

function matchesDomain(url: string, allowedDomains: string[]): boolean {
  const domain = extractDomain(url);
  const allowedDomainsSet = allowedDomains
    .map(d => d.toLowerCase().trim().replace(/^www\./, ''))
    .filter(d => d.length > 0);
  
  return allowedDomainsSet.some(allowedDomain => {
    return domain === allowedDomain || domain.endsWith('.' + allowedDomain);
  });
}

async function webSearch(args: {
  query: string;
  userLocation?: string;
  filters?: { allowedDomains?: string[] | null };
  searchContextSize?: 'low' | 'medium' | 'high';
}): Promise<string> {
  try {
    const client = getFirecrawlClient();
    
    // Map search_context_size to limit
    const searchContextSize = args.searchContextSize || 'medium';
    const limit = mapSearchContextSizeToLimit(searchContextSize);
    
    // Prepare search parameters
    const searchOptions: { limit: number; location?: string } = {
      limit,
    };
    
    // Add location if provided
    if (args.userLocation) {
      searchOptions.location = args.userLocation;
    }
    
    console.log('[webSearch] Starting search', { 
      query: args.query, 
      searchOptions,
      hasFilters: !!args.filters?.allowedDomains 
    });
    
    // Perform search
    const results = await client.search(args.query, searchOptions) as unknown as SearchResponse;
    
    console.log('[webSearch] Raw Firecrawl response', { 
      hasResults: !!results,
      success: results?.success,
      hasData: !!results?.data,
      isArray: Array.isArray(results?.data),
      isObject: results?.data && typeof results?.data === 'object' && !Array.isArray(results?.data),
      dataKeys: results?.data && typeof results?.data === 'object' ? Object.keys(results.data) : [],
      dataLength: Array.isArray(results?.data) ? results.data.length : (results?.data && typeof results.data === 'object' && 'web' in results.data && Array.isArray(results.data.web) ? results.data.web.length : 0),
      error: results?.error,
      warning: results?.warning,
      fullResponseKeys: results ? Object.keys(results) : []
    });
    
    // Check for errors in response
    if (results?.error) {
      console.log('[webSearch] Firecrawl returned an error', { error: results.error });
      return `Error performing web search: ${results.error}`;
    }
    
    // Extract web results - handle both response structures:
    // 1. { data: { web: [...] } } - nested structure (matches Python SDK)
    // 2. { data: [...] } - direct array (matches TypeScript types)
    let webResults: SearchResult[] = [];
    if (results?.data) {
      if (Array.isArray(results.data)) {
        // Direct array structure
        webResults = results.data;
      } else if (typeof results.data === 'object' && 'web' in results.data && Array.isArray(results.data.web)) {
        // Nested structure with web property
        webResults = results.data.web;
      }
    }
    
    console.log('[webSearch] Extracted web results', { 
      count: webResults.length,
      firstResult: webResults[0] || null
    });
    
    // Filter by allowed domains if specified
    let filteredResults = webResults;
    if (args.filters?.allowedDomains && Array.isArray(args.filters.allowedDomains)) {
      const allowedDomains = args.filters.allowedDomains.filter(d => d);
      if (allowedDomains.length > 0) {
        console.log('[webSearch] Filtering by domains', { allowedDomains });
        filteredResults = webResults.filter((result) => {
          const url = result.url || '';
          return matchesDomain(url, allowedDomains);
        });
        console.log('[webSearch] After filtering', { 
          originalCount: webResults.length,
          filteredCount: filteredResults.length 
        });
      }
    }
    
    // Format results
    if (filteredResults.length === 0) {
      console.log('[webSearch] No results found', { 
        hadWebResults: webResults.length > 0,
        wasFiltered: args.filters?.allowedDomains && Array.isArray(args.filters.allowedDomains) && args.filters.allowedDomains.length > 0
      });
      return 'No search results found.';
    }
    
    const formattedResults = filteredResults.map((result, index: number) => {
      const title = result.title || 'No title';
      const url = result.url || '';
      const description = result.description || 'No description available';
      return `${index + 1}. ${title}\n   URL: ${url}\n   ${description}`;
    });
    
    return formattedResults.join('\n\n');
  } catch (error: unknown) {
    if (error instanceof Error && error.message.includes('FIRECRAWL_API_KEY')) {
      throw error;
    }
    const errorMessage = error instanceof Error ? error.message : String(error);
    return `Error performing web search: ${errorMessage}`;
  }
}

const GetWeatherParameters = z.object({
  city: z.string().describe('The city name.'),
});

const HandoffParameters = z.object({
  message: z.string().describe('The message to send to the target agent.'),
  agentId: z.string().describe('The ID of the agent to hand off to (e.g., "weather-assistant", "personal-assistant").'),
});

async function callMcpTool(toolName: string, args: Record<string, unknown>): Promise<string> {
  /** Call a tool on the MCP server.
   *
   * @param toolName - Name of the tool to call
   * @param args - Tool arguments
   * @returns Tool result as string
   * @throws Error if the MCP tool call fails
   */
  const mcpServerUrl = process.env.MCP_SERVER_URL || 'http://localhost:3000/mcp';
  
  const transport = new StreamableHTTPClientTransport(new URL(mcpServerUrl));
  const client = new Client(
    {
      name: 'timestep-agent',
      version: '2026.0.5',
    },
    {
      capabilities: {},
    }
  );

  await client.connect(transport);

  const request: CallToolRequest = {
    method: 'tools/call',
    params: {
      name: toolName,
      arguments: args,
    },
  };

  const result = await client.request(request, CallToolResultSchema);

  // Extract text content from result
  const textParts: string[] = [];
  for (const item of result.content) {
    if (item.type === 'text') {
      textParts.push(item.text);
    }
  }

  await client.close();
  return textParts.join('\n') || 'Tool executed successfully (no text output)';
}

const WebSearchParameters = z.object({
  query: z.string().describe('The search query string.'),
  userLocation: z.string().optional().nullable().describe('Optional location for the search. Lets you customize results to be relevant to a location.'),
  filters: z.object({
    allowedDomains: z.array(z.string()).optional().nullable().describe('Optional list of allowed domains to filter search results.'),
  }).optional().nullable().describe('A filter to apply. Should support \'allowed_domains\' list.'),
  searchContextSize: z.enum(['low', 'medium', 'high']).default('medium').describe('The amount of context to use for the search. One of \'low\', \'medium\', or \'high\'. \'medium\' is the default.'),
});

export async function callFunction(
  name: string,
  args: Record<string, unknown>,
  onApprovalRequired?: (toolCall: ChatCompletionMessageToolCall) => Promise<boolean>,
  sourceContextId?: string,
  onToolResult?: (toolCallId: string, toolName: string, result: string) => void,
  toolCallId?: string,
  onChildMessage?: (message: { kind: string; role: string; messageId: string; parts: Array<{ kind: string; text?: string }>; contextId: string; taskId?: string; toolName?: string; tool_calls?: unknown[] }) => void
): Promise<string> {
  /** Call function that maps tool names to execute functions.
   *
   * @param name - Tool name
   * @param args - Tool arguments
   * @param onApprovalRequired - Optional callback for tool approvals (used by handoff)
   * @param sourceContextId - Optional source context ID for handoff tool
   * @returns Tool result as string
   */
  // Route MCP tools to MCP server
  if (name === 'get_weather') {
    return await callMcpTool(name, args);
  }
  if (name === 'handoff') {
    return await handoffTool(
      { ...(args as { message: string; agentId: string }), sourceContextId },
      {
        onApprovalRequired,
        onChildMessage,
      }
    );
  }
  if (name === 'web_search') {
    return await webSearch(args as {
      query: string;
      userLocation?: string;
      filters?: { allowedDomains?: string[] | null };
      searchContextSize?: 'low' | 'medium' | 'high';
    });
  }
  throw new Error(`Unknown tool: ${name}`);
}

// Export tools and callFunction
export { GetWeatherParameters, WebSearchParameters, HandoffParameters };

