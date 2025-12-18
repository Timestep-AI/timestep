import { Page, FrameLocator } from '@playwright/test';

export function getChatKitFrame(page: Page): FrameLocator {
  return page.frameLocator('iframe').first();
}

export async function waitForChatKit(page: Page, timeout = 30000): Promise<void> {
  await page.waitForSelector('iframe', { timeout, state: 'attached' });
  await page.waitForTimeout(2000);
}

export async function sendMessage(page: Page, message: string): Promise<void> {
  const frame = getChatKitFrame(page);
  const input = frame.locator('textarea').first();
  await input.waitFor({ state: 'visible', timeout: 10000 });
  await input.click();
  await input.fill(message);
  await input.press('Enter');
  await page.waitForTimeout(500);
}

export async function waitForResponse(page: Page, timeout = 30000): Promise<void> {
  const frame = getChatKitFrame(page);
  const startTime = Date.now();
  while (Date.now() - startTime < timeout) {
    const bodyText = await frame.locator('body').textContent();
    if (bodyText && bodyText.length > 50) {
      await page.waitForTimeout(1000);
      return;
    }
    await page.waitForTimeout(500);
  }
  throw new Error(`No response appeared within ${timeout}ms`);
}

export async function setupTest(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForLoadState('domcontentloaded', { timeout: 10000 });
  await waitForChatKit(page);
}

export async function hasMessage(page: Page, text: string): Promise<boolean> {
  const frame = getChatKitFrame(page);
  const bodyText = await frame.locator('body').textContent();
  return bodyText?.toLowerCase().includes(text.toLowerCase()) ?? false;
}

export async function waitForMessage(page: Page, text: string, timeout = 30000): Promise<void> {
  const startTime = Date.now();
  while (Date.now() - startTime < timeout) {
    if (await hasMessage(page, text)) return;
    await page.waitForTimeout(500);
  }
  const frame = getChatKitFrame(page);
  const bodyText = await frame.locator('body').textContent();
  throw new Error(`Message "${text}" not found. Content: ${bodyText?.substring(0, 500)}`);
}

// Cat status helpers
export interface CatStatus {
  energy: number;
  happiness: number;
  cleanliness: number;
}

/**
 * Get the current cat status values from the status panel.
 * The status is displayed as "X / 10" format.
 */
export async function getCatStatus(page: Page): Promise<CatStatus> {
  // Find all status meter elements - they contain "X / 10" text
  const statusTexts = await page.locator('text=/\\d+ \\/ 10/').allTextContents();
  
  // Extract numeric values from "X / 10" format
  const values = statusTexts.map(text => {
    const match = text.match(/(\d+) \/ 10/);
    return match ? parseInt(match[1], 10) : 0;
  });

  // Status order is: Energy, Happiness, Cleanliness (from STATUS_CONFIG)
  return {
    energy: values[0] ?? 0,
    happiness: values[1] ?? 0,
    cleanliness: values[2] ?? 0,
  };
}

/**
 * Click a quick action button by its label.
 */
export async function clickQuickAction(page: Page, label: string): Promise<void> {
  const button = page.locator('button').filter({ hasText: label });
  await button.waitFor({ state: 'visible', timeout: 10000 });
  await button.click();
}

/**
 * Wait for the cat status to change from the initial values.
 * Returns the new status once it changes.
 */
export async function waitForStatusChange(
  page: Page,
  initialStatus: CatStatus,
  timeout = 30000
): Promise<CatStatus> {
  const startTime = Date.now();
  while (Date.now() - startTime < timeout) {
    const currentStatus = await getCatStatus(page);
    if (
      currentStatus.energy !== initialStatus.energy ||
      currentStatus.happiness !== initialStatus.happiness ||
      currentStatus.cleanliness !== initialStatus.cleanliness
    ) {
      return currentStatus;
    }
    await page.waitForTimeout(500);
  }
  throw new Error(`Cat status did not change within ${timeout}ms`);
}

/**
 * Get the cat's name from the status panel.
 */
export async function getCatName(page: Page): Promise<string> {
  const nameElement = page.locator('h2').first();
  return await nameElement.textContent() ?? '';
}
