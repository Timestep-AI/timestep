/**
 * MCP Server using @modelcontextprotocol/sdk.
 * Handles tool execution and handoff sampling.
 *
 * ⚠️ STATUS: INCOMPLETE - PENDING v2 SDK RELEASE
 *
 * This implementation is incomplete because:
 * - v2 of @modelcontextprotocol/sdk (which exports @modelcontextprotocol/server and
 *   @modelcontextprotocol/client) is not yet published to npm
 * - v1.x only exports `Server` from @modelcontextprotocol/sdk/server and doesn't include
 *   the HTTP transport classes we need (StreamableHTTPServerTransport, createMcpExpressApp, McpServer)
 * - Expected v2 release: Q1 2026
 *
 * We explored multiple approaches to use v2 from GitHub:
 * - Git dependencies in imports: Bun's auto-install doesn't support git URLs directly
 * - bun add github:...: Requires package.json, conflicts with inline dependencies
 * - bun create github:...: Clones repo but it's a pnpm workspace requiring build
 * - Direct source imports: Source files have internal dependencies that aren't resolved
 *
 * None of these approaches work with our requirement for inline dependencies without
 * additional files. We must wait for v2 to be published to npm.
 *
 * TODO: When v2 is published, update this file to:
 * 1. Change imports from `@modelcontextprotocol/sdk/server` to `@modelcontextprotocol/server`
 * 2. Use `McpServer` instead of `Server`
 * 3. Use `StreamableHTTPServerTransport` for HTTP transport
 * 4. Use `createMcpExpressApp()` instead of manual Express setup
 * 5. Use `registerTool()` instead of `setRequestHandler()` for tool registration
 * 6. Remove this status comment and implement full functionality
 */

import { randomUUID } from 'node:crypto';
import { Server } from '@modelcontextprotocol/sdk/server';
import { Request, Response } from 'express';
import * as z from 'zod';
import {
  CallToolRequestSchema,
  CallToolResultSchema,
  CallToolResult,
  McpError,
  ErrorCode,
} from '@modelcontextprotocol/sdk';

// Note: v1.x doesn't export StreamableHTTPServerTransport or createMcpExpressApp
// We'll need to use express directly and handle the transport manually
// For now, this is a simplified version that uses Server directly

const MCP_PORT = process.env.MCP_PORT ? parseInt(process.env.MCP_PORT, 10) : 8080;

// Create Express app manually (v1.x doesn't export createMcpExpressApp)
import express from 'express';
const app = express();
app.use(express.json());

// Create MCP server
const server = new Server(
  {
    name: 'MCP Server',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
      logging: {},
    },
  }
);

// Register handoff tool using setRequestHandler
server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
  if (request.params.name === 'handoff') {
    const args = request.params.arguments as { agent_uri?: string; message?: string; context_id?: string };
    
    if (!args.message) {
      throw new McpError(ErrorCode.InvalidParams, 'Message is required for handoff');
    }

    // Use sendRequest to call sampling (this will trigger client's sampling callback)
    try {
      const result = await extra.sendRequest(
        {
          method: 'sampling/createMessage',
          params: {
            messages: [
              {
                role: 'user',
                content: {
                  type: 'text',
                  text: args.message,
                },
              },
            ],
            max_tokens: 1000,
            metadata: { agent_uri: args.agent_uri },
          },
        },
        z.object({
          role: z.literal('assistant'),
          content: z.object({
            type: z.literal('text'),
            text: z.string(),
          }),
          model: z.string().optional(),
        })
      );

      return {
        content: [
          {
            type: 'text',
            text: result.content.text.trim(),
          },
        ],
      } as CallToolResult;
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Handoff failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  } else if (request.params.name === 'get_weather') {
    const args = request.params.arguments as { location?: string };
    const location = args.location || 'Unknown';
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            location: location,
            temperature: '72°F',
            condition: 'Sunny',
            humidity: '65%',
          }),
        },
      ],
    } as CallToolResult;
  } else {
    throw new McpError(ErrorCode.MethodNotFound, `Tool ${request.params.name} not found`);
  }
});

// Simple HTTP handler - v1.x Server doesn't have built-in HTTP transport helpers
// This is a minimal implementation
app.post('/mcp', async (req: Request, res: Response) => {
  try {
    // For v1.x, we'd need to implement the transport layer manually
    // This is a placeholder - the actual implementation would need StreamableHTTPServerTransport
    res.status(501).json({
      jsonrpc: '2.0',
      error: {
        code: -32601,
        message: 'Not implemented: v1.x Server requires manual transport implementation',
      },
      id: req.body?.id ?? null,
    });
  } catch (error) {
    console.error('Error handling MCP request:', error);
    res.status(500).json({
      jsonrpc: '2.0',
      error: {
        code: -32603,
        message: 'Internal server error',
      },
      id: req.body?.id ?? null,
    });
  }
});

if (import.meta.main) {
  app.listen(MCP_PORT, (error) => {
    if (error) {
      console.error('Failed to start server:', error);
      process.exit(1);
    }
    console.log(`MCP Server listening on port ${MCP_PORT} (v1.x - transport not fully implemented)`);
  });
}
