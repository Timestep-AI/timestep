import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.ts'],
    exclude: ['tests/test_cross_language_py_to_ts.ts'], // Called by Python, not run directly
    globals: true,
    fileParallelism: false, // Run tests sequentially to avoid conflicts
    testTimeout: 60000, // 60 seconds for long-running agent tests
    hookTimeout: 60000,
    bail: 1, // Stop on first failure
  },
});

