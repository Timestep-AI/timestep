import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.ts'],
    exclude: [
      'tests/test_cross_language_py_to_ts.ts', // Called by Python, not run directly
      'tests/test_helpers.ts', // Helper file with no test definitions
    ],
    globals: true,
    fileParallelism: false, // Run tests sequentially to avoid conflicts
    testTimeout: 120000, // 2 minutes for queue tests // 60 seconds for long-running agent tests
    hookTimeout: 60000,
    bail: 1, // Stop on first failure
    sequence: {
      hooks: 'stack', // Run hooks in sequence
    },
  },
});

