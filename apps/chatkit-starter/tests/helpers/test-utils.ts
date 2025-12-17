import { Page, FrameLocator } from '@playwright/test';

/**
 * Get the ChatKit iframe frame locator
 */
export function getChatKitFrame(page: Page): FrameLocator {
  return page.frameLocator('iframe').first();
}

/**
 * Wait for ChatKit iframe to be loaded and ready
 */
export async function waitForChatKit(page: Page, timeout: number = 30000): Promise<void> {
  await page.waitForSelector('iframe', { timeout, state: 'attached' });
  await page.waitForTimeout(2000);
}

/**
 * Send a message in the chat input (inside the ChatKit iframe)
 */
export async function sendMessage(page: Page, message: string): Promise<void> {
  const frame = getChatKitFrame(page);
  const input = frame.locator('textarea').first();
  
  await input.waitFor({ state: 'visible', timeout: 10000 });
  await input.click();
  await input.fill(message);
  await input.press('Enter');
  await page.waitForTimeout(500);
}

/**
 * Wait for a response to appear in the chat
 */
export async function waitForResponse(page: Page, timeout: number = 30000): Promise<void> {
  const frame = getChatKitFrame(page);
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    const bodyText = await frame.locator('body').textContent();
    // Look for content that's longer than just the input (AI response)
    if (bodyText && bodyText.length > 50) {
      await page.waitForTimeout(1000);
      return;
    }
    await page.waitForTimeout(500);
  }
  
  throw new Error(`No response appeared within ${timeout}ms`);
}

/**
 * Setup test environment - navigate to app and wait for ChatKit
 */
export async function setupTest(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForLoadState('domcontentloaded', { timeout: 10000 });
  await waitForChatKit(page);
}

/**
 * Check if the chat contains specific text
 */
export async function hasMessage(page: Page, text: string): Promise<boolean> {
  const frame = getChatKitFrame(page);
  const bodyText = await frame.locator('body').textContent();
  return bodyText?.toLowerCase().includes(text.toLowerCase()) ?? false;
}

/**
 * Wait for a message containing specific text to appear
 */
export async function waitForMessage(page: Page, text: string, timeout: number = 30000): Promise<void> {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    if (await hasMessage(page, text)) {
      return;
    }
    await page.waitForTimeout(500);
  }
  
  const frame = getChatKitFrame(page);
  const bodyText = await frame.locator('body').textContent();
  throw new Error(
    `Message containing "${text}" did not appear within ${timeout}ms.\n` +
    `Iframe content: ${bodyText?.substring(0, 500)}`
  );
}
