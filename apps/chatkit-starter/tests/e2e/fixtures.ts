import { test as base } from '@playwright/test';

type TestFixtures = {
  // Add custom fixtures here if needed
};

export const test = base.extend<TestFixtures>({
  // Servers are started/stopped via global setup/teardown in playwright.config.ts
});

export { expect } from '@playwright/test';

