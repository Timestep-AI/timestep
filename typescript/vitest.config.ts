import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.ts'],
    exclude: ['tests/test_cross_language_py_to_ts.ts'], // Called by Python, not run directly
    globals: true,
    fileParallelism: false, // Run tests sequentially to avoid conflicts
  },
});

