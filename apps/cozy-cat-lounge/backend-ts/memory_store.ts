/**
 * Simple in-memory store compatible with the ChatKit Store interface.
 * A production app would implement this using a persistent database.
 */

import type {
  Page,
  ThreadItem,
  ThreadMetadata,
  Attachment,
} from "chatkit-server";
import { NotFoundError, defaultGenerateId } from "chatkit-server";
import type { Store, StoreItemType } from "chatkit-server";

export class MemoryStore implements Store<Record<string, unknown>> {
  private threads: Map<string, ThreadMetadata> = new Map();
  private items: Map<string, ThreadItem[]> = new Map();

  generateThreadId(_context: Record<string, unknown>): string {
    return defaultGenerateId("thread");
  }

  generateItemId(
    itemType: StoreItemType,
    _thread: ThreadMetadata,
    _context: Record<string, unknown>
  ): string {
    return defaultGenerateId(itemType);
  }

  async loadThread(
    threadId: string,
    _context: Record<string, unknown>
  ): Promise<ThreadMetadata> {
    const thread = this.threads.get(threadId);
    if (!thread) {
      throw new NotFoundError(`Thread ${threadId} not found`);
    }
    return thread;
  }

  async saveThread(
    thread: ThreadMetadata,
    _context: Record<string, unknown>
  ): Promise<void> {
    this.threads.set(thread.id, thread);
  }

  async loadThreads(
    limit: number,
    after: string | null,
    order: string,
    _context: Record<string, unknown>
  ): Promise<Page<ThreadMetadata>> {
    const threads = Array.from(this.threads.values());
    return this._paginate(
      threads,
      after,
      limit,
      order,
      (t) => new Date(t.createdAt).getTime(),
      (t) => t.id
    );
  }

  async loadThreadItems(
    threadId: string,
    after: string | null,
    limit: number,
    order: string,
    _context: Record<string, unknown>
  ): Promise<Page<ThreadItem>> {
    const items = this.items.get(threadId) || [];
    return this._paginate(
      items,
      after,
      limit,
      order,
      (i) => new Date(i.createdAt).getTime(),
      (i) => i.id
    );
  }

  async addThreadItem(
    threadId: string,
    item: ThreadItem,
    _context: Record<string, unknown>
  ): Promise<void> {
    if (!this.items.has(threadId)) {
      this.items.set(threadId, []);
    }
    this.items.get(threadId)!.push(item);
  }

  async saveItem(
    threadId: string,
    item: ThreadItem,
    _context: Record<string, unknown>
  ): Promise<void> {
    const items = this.items.get(threadId) || [];
    const idx = items.findIndex((existing) => existing.id === item.id);
    if (idx >= 0) {
      items[idx] = item;
    } else {
      items.push(item);
    }
    this.items.set(threadId, items);
  }

  async loadItem(
    threadId: string,
    itemId: string,
    _context: Record<string, unknown>
  ): Promise<ThreadItem> {
    const items = this.items.get(threadId) || [];
    const item = items.find((i) => i.id === itemId);
    if (!item) {
      throw new NotFoundError(`Item ${itemId} not found in thread ${threadId}`);
    }
    return item;
  }

  async deleteThread(
    threadId: string,
    _context: Record<string, unknown>
  ): Promise<void> {
    this.threads.delete(threadId);
    this.items.delete(threadId);
  }

  async deleteThreadItem(
    threadId: string,
    itemId: string,
    _context: Record<string, unknown>
  ): Promise<void> {
    const items = this.items.get(threadId) || [];
    this.items.set(
      threadId,
      items.filter((item) => item.id !== itemId)
    );
  }

  // Attachments are not implemented in the quickstart store
  async saveAttachment(
    _attachment: Attachment,
    _context: Record<string, unknown>
  ): Promise<void> {
    throw new Error("Attachments not implemented");
  }

  async loadAttachment(
    _attachmentId: string,
    _context: Record<string, unknown>
  ): Promise<Attachment> {
    throw new Error("Attachments not implemented");
  }

  async deleteAttachment(
    _attachmentId: string,
    _context: Record<string, unknown>
  ): Promise<void> {
    throw new Error("Attachments not implemented");
  }

  private _paginate<T>(
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
      for (let idx = 0; idx < sortedRows.length; idx++) {
        if (cursorKey(sortedRows[idx]) === after) {
          start = idx + 1;
          break;
        }
      }
    }

    const data = sortedRows.slice(start, start + limit);
    const hasMore = start + limit < sortedRows.length;
    const nextAfter = hasMore && data.length > 0 ? cursorKey(data[data.length - 1]) : null;

    return { data, hasMore, after: nextAfter };
  }
}

