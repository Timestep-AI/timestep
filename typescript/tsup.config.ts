import { defineConfig } from 'tsup';

// List of Node.js built-in modules that should be externalized
const nodeBuiltins = [
  'events',
  'stream',
  'util',
  'crypto',
  'http',
  'https',
  'net',
  'tls',
  'url',
  'path',
  'fs',
  'os',
  'buffer',
  'process',
  'zlib',
  'querystring',
  'assert',
  'child_process',
  'cluster',
  'dgram',
  'dns',
  'domain',
  'module',
  'perf_hooks',
  'punycode',
  'readline',
  'repl',
  'string_decoder',
  'sys',
  'timers',
  'tty',
  'v8',
  'vm',
  'worker_threads',
];

export default defineConfig({
  entry: ['timestep/index.ts'],
  format: ['esm'],
  dts: true,
  sourcemap: true,
  splitting: false,
  // Externalize Node.js built-ins and dependencies
  external: [
    // Node.js built-ins (tsup should handle these automatically, but being explicit)
    ...nodeBuiltins,
    /^node:/,
    // External dependencies from package.json
    'openai',
    'ollama',
    '@mendable/firecrawl-js',
    '@dbos-inc/dbos-sdk',
    'postgres',
    'pg',
    'ms',
    'debug',
    '@modelcontextprotocol/sdk',
    // Externalize ws package - it uses Node.js built-ins and should not be bundled
    // This is a transitive dependency from @openai/agents-realtime
    'ws',
    /^ws$/,
  ],
  // Don't bundle dependencies that use Node.js built-ins
  // This ensures ws and other packages that require Node.js built-ins are not bundled
  noExternal: [],
  // Target Node.js environment
  platform: 'node',
});

