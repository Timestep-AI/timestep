import { test, expect } from './fixtures';
import { setupTest, sendMessage, waitForResponse } from '../helpers/test-utils';

test.describe('Cozy Cat Lounge', () => {
  test('should load the app', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 30000 });
    expect(page.url()).toContain('localhost:3000');
  });

  test('should initialize ChatKit', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 30000 });
    const body = await page.locator('body').textContent();
    expect(body).toBeTruthy();
  });

  test('should send a message and receive a response', async ({ page }) => {
    await setupTest(page);
    await page.waitForTimeout(3000);
    await sendMessage(page, 'Hello, this is a test message');
    await waitForResponse(page, 30000);
  });

  test('should handle page navigation', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 30000 });
    const title = await page.title();
    expect(title).toBeTruthy();
  });
});
