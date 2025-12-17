import { test, expect } from './fixtures';
import { setupTest, sendMessage, waitForResponse, waitForMessage } from '../helpers/test-utils';

test.describe('ChatKit Starter', () => {
  test('should load the app', async ({ page }) => {
    await page.goto('/');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle', { timeout: 60000 });
    
    // Check that the page loaded successfully
    expect(page.url()).toContain('localhost:3000');
  });

  test('should initialize ChatKit', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 60000 });
    
    // ChatKit should be initialized - check for common elements
    // The page should have some ChatKit-related content
    const body = await page.locator('body').textContent();
    expect(body).toBeTruthy();
  });

  test('should send a message and receive a response', async ({ page }) => {
    await setupTest(page);
    
    // Wait for ChatKit to be fully ready
    await page.waitForTimeout(3000);
    
    // Send a message - this MUST work for the test to be meaningful
    await sendMessage(page, 'Hello, this is a test message');
    
    // Wait for a response from the AI
    await waitForResponse(page, 30000);
  });

  test('should maintain conversation history across multiple messages', async ({ page }) => {
    // Increase timeout for this specific test
    test.setTimeout(60000);
    
    await setupTest(page);
    
    // Wait for ChatKit to be fully ready
    await page.waitForTimeout(2000);
    
    // First message: "What's 2+2?"
    await sendMessage(page, "What's 2+2?");
    
    // Wait for response containing "4" - LLM might write it differently
    await waitForMessage(page, '4', 20000);
    
    // Wait a bit between messages
    await page.waitForTimeout(1000);
    
    // Second message: "and that times three?"
    await sendMessage(page, 'and that times three?');
    
    // Wait for response containing "12" (verifying history works - it should know 4 * 3 = 12)
    await waitForMessage(page, '12', 20000);
  });

  test('should handle page navigation', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 60000 });
    
    // Page should be accessible
    const title = await page.title();
    expect(title).toBeTruthy();
  });
});

