/** CLI entry point for MCP server. */

import { runServer } from './server.js';

function parseArgs(): { host: string; port: number } {
  const args = process.argv.slice(2);
  let host = process.env.MCP_HOST || '0.0.0.0';
  let port = process.env.MCP_PORT ? parseInt(process.env.MCP_PORT, 10) : 3000;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--host' && i + 1 < args.length) {
      host = args[++i];
    } else if (arg === '--port' && i + 1 < args.length) {
      port = parseInt(args[++i], 10);
      if (isNaN(port)) {
        console.error('Error: --port must be a number');
        process.exit(1);
      }
    } else if (arg === '--help' || arg === '-h') {
      console.log(`
Usage: tsx timestep/mcp/cli.ts [options]

Options:
  --host <address>    Host address to bind to (default: 0.0.0.0)
  --port <number>     Port number to listen on (default: 3000)
  --help, -h          Show this help message

Environment variables:
  MCP_HOST            Host address (overridden by --host)
  MCP_PORT            Port number (overridden by --port)
`);
      process.exit(0);
    } else {
      console.error(`Unknown option: ${arg}`);
      console.error('Use --help for usage information');
      process.exit(1);
    }
  }

  return { host, port };
}

function main(): void {
  const { host, port } = parseArgs();

  console.log(`Starting Timestep MCP server on ${host}:${port}`);
  console.log(`MCP endpoint: http://${host}:${port}/mcp`);

  runServer(host, port);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { main };

