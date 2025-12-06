/** Common tools for agents, including web search using Firecrawl. */

import { tool } from '@openai/agents';
import Firecrawl from '@mendable/firecrawl-js';

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

export const webSearch = tool({
  name: 'web_search',
  description: 'A tool that lets the LLM search the web using Firecrawl.',
  parameters: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'The search query string.',
      },
      userLocation: {
        type: 'string',
        description: 'Optional location for the search. Lets you customize results to be relevant to a location.',
      },
      filters: {
        type: 'object',
        properties: {
          allowedDomains: {
            type: 'array',
            items: { type: 'string' },
            description: 'Optional list of allowed domains to filter search results.',
          },
        },
        additionalProperties: false,
      },
      searchContextSize: {
        type: 'string',
        enum: ['low', 'medium', 'high'],
        description: 'The amount of context to use for the search. One of "low", "medium", or "high". "medium" is the default.',
        default: 'medium',
      },
    },
    required: ['query'],
    additionalProperties: false,
  } as any,
  execute: async (args: any): Promise<string> => {
    try {
      const client = getFirecrawlClient();
      
      // Map search_context_size to limit
      const searchContextSize = args.searchContextSize || 'medium';
      const limit = mapSearchContextSizeToLimit(searchContextSize);
      
      // Prepare search parameters
      const searchOptions: any = {
        limit,
      };
      
      // Add location if provided
      if (args.userLocation) {
        searchOptions.location = args.userLocation;
      }
      
      // Perform search
      const results: any = await client.search(args.query, searchOptions);
      
      // Extract web results
      const webResults = results?.data?.web || [];
      
      // Filter by allowed domains if specified
      let filteredResults = webResults;
      if (args.filters?.allowedDomains && Array.isArray(args.filters.allowedDomains)) {
        const allowedDomains = args.filters.allowedDomains.filter(d => d);
        if (allowedDomains.length > 0) {
          filteredResults = webResults.filter((result: any) => {
            const url = result.url || '';
            return matchesDomain(url, allowedDomains);
          });
        }
      }
      
      // Format results
      if (filteredResults.length === 0) {
        return 'No search results found.';
      }
      
      const formattedResults = filteredResults.map((result: any, index: number) => {
        const title = result.title || 'No title';
        const url = result.url || '';
        const description = result.description || 'No description available';
        return `${index + 1}. ${title}\n   URL: ${url}\n   ${description}`;
      });
      
      return formattedResults.join('\n\n');
    } catch (error: any) {
      if (error instanceof Error && error.message.includes('FIRECRAWL_API_KEY')) {
        throw error;
      }
      return `Error performing web search: ${error?.message || String(error)}`;
    }
  },
});

