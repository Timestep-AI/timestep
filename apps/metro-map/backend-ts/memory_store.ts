/**
 * Simple in-memory store compatible with the ChatKit Store interface.
 */

import type { Store, Page, ThreadItem, ThreadMetadata, Attachment, StoreItemType } from "chatkit-server";

export class NotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NotFoundError";
  }
}

export class MemoryStore implements Store<Record<string, unknown>> {
  private threads: Map<string, ThreadMetadata> = new Map();
  private items: Map<string, ThreadItem[]> = new Map();
  private threadCounter = 0;
  private itemCounter = 0;

  async loadThread(threadId: string, _context: Record<string, unknown>): Promise<ThreadMetadata> {
    const thread = this.threads.get(threadId);
    if (!thread) throw new NotFoundError(`Thread ${threadId} not found`);
    return thread;
  }

  async saveThread(thread: ThreadMetadata, _context: Record<string, unknown>): Promise<void> {
    this.threads.set(thread.id, thread);
  }

  async loadThreads(
    limit: number,
    after: string | null,
    order: string,
    _context: Record<string, unknown>
  ): Promise<Page<ThreadMetadata>> {
    const threads = Array.from(this.threads.values());
    return this.paginate(threads, after, limit, order, (t) => new Date(t.created_at).getTime(), (t) => t.id);
  }

  async loadThreadItems(
    threadId: string,
    after: string | null,
    limit: number,
    order: string,
    _context: Record<string, unknown>
  ): Promise<Page<ThreadItem>> {
    const items = this.items.get(threadId) || [];
    return this.paginate(items, after, limit, order, (i) => new Date(i.created_at).getTime(), (i) => i.id);
  }

  async addThreadItem(threadId: string, item: ThreadItem, _context: Record<string, unknown>): Promise<void> {
    if (!this.items.has(threadId)) this.items.set(threadId, []);
    this.items.get(threadId)!.push(item);
  }

  async saveItem(threadId: string, item: ThreadItem, _context: Record<string, unknown>): Promise<void> {
    const items = this.items.get(threadId) || [];
    const idx = items.findIndex((i) => i.id === item.id);
    if (idx >= 0) items[idx] = item;
    else items.push(item);
    this.items.set(threadId, items);
  }

  async loadItem(threadId: string, itemId: string, _context: Record<string, unknown>): Promise<ThreadItem> {
    const items = this.items.get(threadId) || [];
    const item = items.find((i) => i.id === itemId);
    if (!item) throw new NotFoundError(`Item ${itemId} not found in thread ${threadId}`);
    return item;
  }

  async deleteThread(threadId: string, _context: Record<string, unknown>): Promise<void> {
    this.threads.delete(threadId);
    this.items.delete(threadId);
  }

  async deleteThreadItem(threadId: string, itemId: string, _context: Record<string, unknown>): Promise<void> {
    const items = this.items.get(threadId);
    if (items) this.items.set(threadId, items.filter((i) => i.id !== itemId));
  }

  async saveAttachment(_attachment: Attachment, _context: Record<string, unknown>): Promise<void> {
    throw new Error("Attachments not implemented");
  }

  async loadAttachment(_attachmentId: string, _context: Record<string, unknown>): Promise<Attachment> {
    throw new Error("Attachments not implemented");
  }

  async deleteAttachment(_attachmentId: string, _context: Record<string, unknown>): Promise<void> {
    throw new Error("Attachments not implemented");
  }

  generateThreadId(_context: Record<string, unknown>): string {
    this.threadCounter++;
    return `thr_${this.threadCounter.toString(16).padStart(8, "0")}`;
  }

  generateItemId(_itemType: StoreItemType, _thread: ThreadMetadata, _context: Record<string, unknown>): string {
    this.itemCounter++;
    return `item_${this.itemCounter.toString(16).padStart(8, "0")}`;
  }

  private paginate<T>(
    rows: T[],
    after: string | null,
    limit: number,
    order: string,
    sortKey: (item: T) => number,
    cursorKey: (item: T) => string
  ): Page<T> {
    const sortedRows = [...rows].sort((a, b) => {
      const diff = sortKey(a) - sortKey(b);
      return order === "desc" ? -diff : diff;
    });

    let start = 0;
    if (after) {
      for (let i = 0; i < sortedRows.length; i++) {
        if (cursorKey(sortedRows[i]) === after) {
          start = i + 1;
          break;
        }
      }
    }

    const data = sortedRows.slice(start, start + limit);
    const hasMore = start + limit < sortedRows.length;
    const nextAfter = hasMore && data.length > 0 ? cursorKey(data[data.length - 1]) : null;

    return { data, has_more: hasMore, after: nextAfter };
  }
}

