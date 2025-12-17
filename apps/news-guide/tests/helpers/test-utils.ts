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
