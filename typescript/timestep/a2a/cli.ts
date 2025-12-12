/** CLI entry point for A2A server. */

import { runServer } from './server.js';

function parseArgs(): { host: string; port: number; model: string } {
  const args = process.argv.slice(2);
  let host = process.env.HOST || '0.0.0.0';
  let port = process.env.PORT ? parseInt(process.env.PORT, 10) : 8080;
  let model = process.env.MODEL || 'gpt-4.1';

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
    } else if (arg === '--model' && i + 1 < args.length) {
      model = args[++i];
    } else if (arg === '--help' || arg === '-h') {
      console.log(`
Usage: tsx timestep/a2a/cli.ts [options]

Options:
  --host <address>    Host address to bind to (default: 0.0.0.0)
  --port <number>     Port number to listen on (default: 8080)
  --model <name>      OpenAI model name to use (default: gpt-4.1)
  --help, -h          Show this help message

Environment variables:
  HOST                Host address (overridden by --host)
  PORT                Port number (overridden by --port)
  MODEL               Model name (overridden by --model)
`);
      process.exit(0);
    } else {
      console.error(`Unknown option: ${arg}`);
      console.error('Use --help for usage information');
      process.exit(1);
    }
  }

  return { host, port, model };
}

function main(): void {
  const { host, port, model } = parseArgs();

  console.log(`Starting Timestep A2A server on ${host}:${port}`);
  console.log(`Agent Card: http://${host}:${port}/.well-known/agent-card.json`);

  runServer(host, port, undefined, model);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

