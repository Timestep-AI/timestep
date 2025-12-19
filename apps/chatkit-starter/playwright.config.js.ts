import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for testing the JS (Deno) backend.
 * Uses the same tests as the Python backend but with different server startup.
 */
export default defineConfig({
  testDir: './tests/e2e',
  /* Global setup and teardown - uses JS backend variants */
  globalSetup: require.resolve('./tests/helpers/global-setup-js.ts'),
  globalTeardown: require.resolve('./tests/helpers/global-teardown-js.ts'),
  /* Maximum time the entire test run can take (60 seconds) */
  globalTimeout: 60000,
  /* Maximum time one test can run for */
  timeout: 25000,
  /* Maximum time to wait for assertion */
  expect: {
    timeout: 10000,
  },
  /* Run tests in files in parallel */
  fullyParallel: false, // Disable parallel to avoid port conflicts
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: 'html',
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: 'http://localhost:3000',
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
    /* Screenshot on failure */
    screenshot: 'only-on-failure',
    /* Video on failure */
    video: 'retain-on-failure',
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});

