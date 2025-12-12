/** MCP server implementation for Timestep tools. */

import type { Request, Response } from 'express';
import type { Application } from 'express';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { createMcpExpressApp } from '@modelcontextprotocol/sdk/server/express.js';
import * as z from 'zod/v4';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';

function getServer(): McpServer {
  /** Create and configure the MCP server with tools. */
  const server = new McpServer(
    {
      name: 'timestep-mcp-server',
      version: '2026.0.5',
    },
    { capabilities: { logging: {} } }
  );

  // Register get_weather tool
  server.tool(
    'get_weather',
    'Returns weather info for the specified city.',
    {
      city: z.string().describe('The city name.'),
    },
    async ({ city }): Promise<CallToolResult> => {
      // Simple mock implementation - in production this would call a real weather API
      return {
        content: [
          {
            type: 'text',
            text: `The weather in ${city} is sunny`,
          },
        ],
      };
    }
  );


  return server;
}

export function createServer(host: string = '0.0.0.0', port: number = 3000): Application {
  /** Create and configure the Express app with MCP endpoints.
   *
   * @param host - Host address to bind to.
   * @param port - Port number to listen on.
   * @returns Configured Express application.
   */
  // Allow Docker service names and localhost for development
  // Include both with and without port for flexibility
  const defaultAllowedHosts = [
    'localhost',
    `localhost:${port}`,
    '127.0.0.1',
    `127.0.0.1:${port}`,
    'mcp-server',
    `mcp-server:${port}`,
    'a2a-server',
    `a2a-server:${port}`,
    '[::1]',
    `[::1]:${port}`,
  ];
  
  // Parse allowed hosts from environment or use defaults
  // Always include port variations for each host
  const allowedHosts = process.env.MCP_ALLOWED_HOSTS
    ? process.env.MCP_ALLOWED_HOSTS.split(',').flatMap((host) => {
        const trimmed = host.trim();
        return [trimmed, `${trimmed}:${port}`];
      })
    : defaultAllowedHosts;
  
  // Create a singleton server instance that will be reused
  const mcpServer = getServer();
  
  const app = createMcpExpressApp({
    host,
    allowedHosts,
  });
  
  // Override the default handler to use our singleton server
  app.post('/mcp', async (req: Request, res: Response) => {
    try {
      const transport: StreamableHTTPServerTransport = new StreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
        allowedHosts,
      });
      await mcpServer.connect(transport);
      await transport.handleRequest(req, res, req.body);
      res.on('close', () => {
        console.log('MCP request closed');
        transport.close();
        // Don't close the server, just the transport
      });
    } catch (error) {
      console.error('Error handling MCP request:', error);
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: '2.0',
          error: {
            code: -32603,
            message: 'Internal server error',
          },
          id: null,
        });
      }
    }
  });

  app.get('/mcp', async (_req: Request, res: Response) => {
    res.writeHead(405).end(
      JSON.stringify({
        jsonrpc: '2.0',
        error: {
          code: -32000,
          message: 'Method not allowed.',
        },
        id: null,
      })
    );
  });

  app.delete('/mcp', async (_req: Request, res: Response) => {
    res.writeHead(405).end(
      JSON.stringify({
        jsonrpc: '2.0',
        error: {
          code: -32000,
          message: 'Method not allowed.',
        },
        id: null,
      })
    );
  });

  return app;
}

export function runServer(host: string = '0.0.0.0', port: number = 3000): void {
  /** Run the MCP server.
   *
   * @param host - Host address to bind to.
   * @param port - Port number to listen on.
   */
  const app = createServer(host, port);
  const PORT = process.env.MCP_PORT ? parseInt(process.env.MCP_PORT, 10) : port;
  const HOST = process.env.MCP_HOST || host;

  app.listen(PORT, HOST, (error?: Error) => {
    if (error) {
      console.error('Failed to start MCP server:', error);
      process.exit(1);
    }
    console.log(`[Timestep MCP] Server started on http://${HOST}:${PORT}`);
    console.log('[Timestep MCP] Press Ctrl+C to stop the server');
  });

  // Handle server shutdown
  process.on('SIGINT', async () => {
    console.log('Shutting down MCP server...');
    process.exit(0);
  });
}

