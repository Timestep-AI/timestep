# Tools

Timestep provides built-in tools that can be used with agents. Currently, the main tool is `web_search` / `webSearch` for performing web searches using Firecrawl.

## web_search / webSearch

A tool that enables agents to search the web using Firecrawl. This tool is useful for agents that need to access current information or search for specific content.

### Function Signature

=== "Python"

    ```python
    @function_tool
    def web_search(
        query: str,
        user_location: Optional[str] = None,
        filters: Optional[Any] = None,
        search_context_size: Literal["low", "medium", "high"] = "medium",
    ) -> str
    ```

=== "TypeScript"

    ```typescript
    export const webSearch: Tool
    ```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `string` | The search query string. **Required.** |
| `user_location` / `userLocation` | `string \| undefined` | Optional location for the search. Customizes results to be relevant to a specific location. |
| `filters` | `object \| undefined` | Optional filter object. Supports `allowed_domains` / `allowedDomains` array to filter results by domain. |
| `search_context_size` / `searchContextSize` | `"low" \| "medium" \| "high"` | The amount of context to use for the search. Defaults to `"medium"`. |

#### Filter Object

The `filters` parameter supports:

| Property | Type | Description |
|----------|------|-------------|
| `allowed_domains` / `allowedDomains` | `string[] \| null` | Array of allowed domains. Only results from these domains will be returned. |

#### Search Context Size

| Value | Description | Limit |
|-------|-------------|-------|
| `"low"` | Minimal context | 5 results |
| `"medium"` | Moderate context | 10 results |
| `"high"` | Maximum context | 20 results |

### Returns

| Type | Description |
|------|-------------|
| `string` | A formatted string containing search results with titles, URLs, and descriptions. Returns `"No search results found."` if no results are available. |

### Prerequisites

To use the web search tool, you need:

1. **Firecrawl API Key**: Set the `FIRECRAWL_API_KEY` environment variable
2. **Firecrawl Package**: The `firecrawl-py` (Python) or `@mendable/firecrawl-js` (TypeScript) package is required

### Example Usage

=== "Python"

    ```python
    from timestep import web_search
    from agents import Agent, Runner, RunConfig
    from timestep import MultiModelProvider
    import os

    # Create agent with web search tool
    agent = Agent(
        model="gpt-4",
        tools=[web_search]
    )

    model_provider = MultiModelProvider(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    run_config = RunConfig(model_provider=model_provider)
    result = await Runner.run(
        agent,
        [{"role": "user", "content": "What's the latest news about AI?"}],
        run_config=run_config
    )
    ```

=== "TypeScript"

    ```typescript
    import { webSearch } from '@timestep-ai/timestep';
    import { Agent, Runner } from '@openai/agents';
    import { MultiModelProvider } from '@timestep-ai/timestep';

    // Create agent with web search tool
    const agent = new Agent({
      model: 'gpt-4',
      tools: [webSearch],
    });

    const modelProvider = new MultiModelProvider({
      openai_api_key: Deno.env.get('OPENAI_API_KEY') || '',
    });

    const runner = new Runner({ modelProvider });
    const result = await runner.run(agent, [
      { role: 'user', content: "What's the latest news about AI?" },
    ]);
    ```

### Advanced Examples

#### With Location Filter

=== "Python"

    ```python
    # The agent can call web_search with location
    # Example: web_search(query="restaurants", user_location="San Francisco, CA")
    ```

=== "TypeScript"

    ```typescript
    // The agent can call webSearch with location
    // Example: webSearch({ query: "restaurants", userLocation: "San Francisco, CA" })
    ```

#### With Domain Filter

=== "Python"

    ```python
    # The agent can call web_search with domain filters
    # Example: web_search(
    #     query="Python tutorials",
    #     filters={"allowed_domains": ["python.org", "realpython.com"]}
    # )
    ```

=== "TypeScript"

    ```typescript
    // The agent can call webSearch with domain filters
    // Example: webSearch({
    //   query: "Python tutorials",
    //   filters: { allowedDomains: ["python.org", "realpython.com"] }
    // })
    ```

#### With Custom Context Size

=== "Python"

    ```python
    # The agent can call web_search with custom context size
    # Example: web_search(query="research papers", search_context_size="high")
    ```

=== "TypeScript"

    ```typescript
    // The agent can call webSearch with custom context size
    // Example: webSearch({ query: "research papers", searchContextSize: "high" })
    ```

### Error Handling

If the `FIRECRAWL_API_KEY` environment variable is not set, the tool will raise an error:

=== "Python"

    ```python
    # Raises ValueError if FIRECRAWL_API_KEY is not set
    result = web_search(query="test")
    ```

=== "TypeScript"

    ```typescript
    // Throws Error if FIRECRAWL_API_KEY is not set
    const result = await webSearch.execute({ query: 'test' });
    ```

If a search error occurs (other than missing API key), the tool returns an error message string:

=== "Python"

    ```python
    # Returns error message string on failure
    result = web_search(query="test")
    # Result might be: "Error performing web search: <error message>"
    ```

=== "TypeScript"

    ```typescript
    // Returns error message string on failure
    const result = await webSearch.execute({ query: 'test' });
    // Result might be: "Error performing web search: <error message>"
    ```

### Response Format

The tool returns a formatted string with search results:

```
1. Title of Result
   URL: https://example.com
   Description of the result

2. Another Title
   URL: https://another-example.com
   Another description
```

If no results are found, it returns: `"No search results found."`

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `FIRECRAWL_API_KEY` | Your Firecrawl API key | Yes |

### Installation

The web search tool requires the Firecrawl package:

=== "Python"

    ```bash
    pip install firecrawl-py
    ```

=== "TypeScript"

    ```bash
    npm install @mendable/firecrawl-js
    # or
    pnpm add @mendable/firecrawl-js
    ```

Note: The Firecrawl package is typically included as a dependency of Timestep, so you may not need to install it separately.

### See Also

- [Use Cases](../use-cases.md) - For examples of using tools with agents
- [Getting Started](../getting-started.md) - For basic agent setup

