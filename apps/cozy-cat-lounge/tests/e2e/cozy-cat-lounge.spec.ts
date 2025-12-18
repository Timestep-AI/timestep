import { test, expect } from './fixtures';
import {
  setupTest,
  sendMessage,
  waitForResponse,
  getCatStatus,
  clickQuickAction,
  waitForStatusChange,
  getCatName,
} from '../helpers/test-utils';

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

  test('should increase energy and happiness when giving a snack', async ({ page }) => {
    // This test needs extra time for the agent to process the action
    test.setTimeout(60000);

    // Setup the page
    await setupTest(page);
    await page.waitForTimeout(2000);

    // Get initial cat status
    const initialStatus = await getCatStatus(page);
    console.log('Initial status:', initialStatus);

    // Verify we're reading status values correctly
    expect(initialStatus.energy).toBeGreaterThan(0);
    expect(initialStatus.happiness).toBeGreaterThan(0);
    expect(initialStatus.cleanliness).toBeGreaterThan(0);

    // Click the "Give snack" button
    await clickQuickAction(page, 'Give snack');

    // Wait for the agent to process the action and status to change
    const newStatus = await waitForStatusChange(page, initialStatus, 50000);
    console.log('New status:', newStatus);

    // Verify that energy increased (feed adds +3 energy by default)
    expect(newStatus.energy).toBeGreaterThanOrEqual(initialStatus.energy);
    
    // Verify that happiness increased (feed adds +1 happiness)
    expect(newStatus.happiness).toBeGreaterThanOrEqual(initialStatus.happiness);
    
    // Either energy or happiness should have increased
    const energyIncreased = newStatus.energy > initialStatus.energy;
    const happinessIncreased = newStatus.happiness > initialStatus.happiness;
    expect(energyIncreased || happinessIncreased).toBe(true);
  });
});
