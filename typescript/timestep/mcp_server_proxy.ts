import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { tool } from '@openai/agents';
import { z } from 'zod';

/**
 * Built-in MCP tool definition
 */
export interface BuiltInMcpTool {
  name: string;
  description: string;
  inputSchema: any;
  needsApproval?: boolean;
  execute: (params: any) => Promise<string>;
}

/**
 * Built-in tools available in the local MCP server
 */
const BUILT_IN_TOOLS: BuiltInMcpTool[] = [
  {
    name: 'get_weather',
    description: 'Get the weather for a given city',
    inputSchema: {
      type: 'object',
      properties: {
        city: { type: 'string' },
      },
      required: ['city'],
    },
    needsApproval: true,
    async execute({ city }) {
      return `The weather in ${city} is sunny.`;
    },
  },
];

/**
 * Fetches tools from an MCP server and/or includes built-in tools.
 *
 * @param serverUrl - The URL of the MCP server (e.g., 'https://gitmcp.io/timestep-ai/timestep'), or null for built-in only
 * @param includeBuiltIn - Whether to include built-in tools from the local MCP server
 * @param requireApproval - Configuration for which tools require approval
 * @returns Array of agent tools
 */
export async function fetchMcpTools(
  serverUrl: string | null,
  includeBuiltIn: boolean = false,
  requireApproval?: {
    never?: { toolNames: string[] };
    always?: { toolNames: string[] };
  }
) {
  let agentTools: any[] = [];

  // Fetch remote tools if serverUrl is provided
  if (serverUrl) {
    // Create MCP client
    const client = new Client(
      {
        name: 'timestep-mcp-client',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    // Create SSE transport
    const transport = new SSEClientTransport(new URL(serverUrl));

    // Connect to the server
    await client.connect(transport);

    // List all available tools from the MCP server
    const toolsResponse = await client.listTools();
    const mcpTools = toolsResponse.tools || [];

    // Close the connection after fetching tools
    await client.close();

    // Convert MCP tools to OpenAI agent tools
    const remoteTools = mcpTools.map((mcpTool) => {
      // Determine if this tool needs approval
      const neverRequireApproval = requireApproval?.never?.toolNames.includes(mcpTool.name) ?? false;
      const alwaysRequireApproval = requireApproval?.always?.toolNames.includes(mcpTool.name) ?? false;
      const needsApproval = alwaysRequireApproval || (!neverRequireApproval && false); // Default to false

      // Convert JSON schema to Zod schema
      const parameters = mcpTool.inputSchema || { type: 'object', properties: {} };
      const zodSchema = jsonSchemaToZod(parameters);

      return tool({
        name: mcpTool.name,
        description: mcpTool.description || '',
        parameters: zodSchema,
        needsApproval,
        async execute(params) {
          // Open a new connection for each tool execution
          const execClient = new Client(
            {
              name: 'timestep-mcp-client',
              version: '1.0.0',
            },
            {
              capabilities: {
                tools: {},
              },
            }
          );

          const execTransport = new SSEClientTransport(new URL(serverUrl));
          await execClient.connect(execTransport);

          try {
            // Call the MCP tool
            const result = await execClient.callTool({
              name: mcpTool.name,
              arguments: params,
            });

            // Return the result
            if (result.isError) {
              throw new Error(`MCP tool error: ${JSON.stringify(result.content)}`);
            }

            // Extract text content from result
            const textContent = result.content
              .filter((item: any) => item.type === 'text')
              .map((item: any) => item.text)
              .join('\n');

            return textContent || JSON.stringify(result.content);
          } finally {
            // Always close the connection after execution
            await execClient.close();
          }
        },
      });
    });

    agentTools.push(...remoteTools);
  }

  // Add built-in tools if requested
  if (includeBuiltIn) {
    const localTools = BUILT_IN_TOOLS.map((builtInTool) => {
      // Determine if this tool needs approval (apply requireApproval config)
      const neverRequireApproval = requireApproval?.never?.toolNames.includes(builtInTool.name) ?? false;
      const alwaysRequireApproval = requireApproval?.always?.toolNames.includes(builtInTool.name) ?? false;
      const needsApproval = alwaysRequireApproval || (!neverRequireApproval && (builtInTool.needsApproval ?? false));

      // Convert JSON schema to Zod schema
      const parameters = builtInTool.inputSchema || { type: 'object', properties: {} };
      const zodSchema = jsonSchemaToZod(parameters);

      return tool({
        name: builtInTool.name,
        description: builtInTool.description,
        parameters: zodSchema,
        needsApproval,
        async execute(params) {
          return await builtInTool.execute(params);
        },
      });
    });

    agentTools.push(...localTools);
  }

  return agentTools;
}

/**
 * Converts a JSON Schema to a Zod schema.
 * This is a simplified conversion that handles common cases.
 */
function jsonSchemaToZod(schema: any): z.ZodTypeAny {
  if (schema.type === 'object') {
    const shape: Record<string, z.ZodTypeAny> = {};
    const properties = schema.properties || {};
    const required = schema.required || [];

    for (const [key, value] of Object.entries(properties)) {
      let fieldSchema = jsonSchemaToZod(value);

      // Make field optional AND nullable if not required (required by OpenAI API)
      if (!required.includes(key)) {
        fieldSchema = fieldSchema.nullable().optional();
      }

      shape[key] = fieldSchema;
    }

    return z.object(shape);
  } else if (schema.type === 'string') {
    return z.string();
  } else if (schema.type === 'number') {
    return z.number();
  } else if (schema.type === 'integer') {
    return z.number().int();
  } else if (schema.type === 'boolean') {
    return z.boolean();
  } else if (schema.type === 'array') {
    const itemSchema = schema.items ? jsonSchemaToZod(schema.items) : z.any();
    return z.array(itemSchema);
  } else {
    // Fallback to any for unknown types
    return z.any();
  }
}
