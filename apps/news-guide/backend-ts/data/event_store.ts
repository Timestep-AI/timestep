/**
 * Data models and search helpers for News Guide events.
 */

import * as path from "node:path";

export interface EventRecord {
  id: string;
  date: string;
  dayOfWeek: string;
  time: string;
  location: string;
  title: string;
  details: string;
  category: string;
  keywords: string[];
}

export class EventStore {
  private dataDir: string;
  private events: Map<string, EventRecord> = new Map();
  private order: string[] = [];

  constructor(dataDir: string) {
    this.dataDir = dataDir;
    this.reload();
  }

  get eventsPath(): string {
    return path.join(this.dataDir, "events.json");
  }

  reload(): void {
    const raw = JSON.parse(Deno.readTextFileSync(this.eventsPath));
    if (!Array.isArray(raw)) {
      throw new Error("events.json must contain a list of event entries.");
    }

    this.events.clear();
    this.order = [];

    for (const entry of raw) {
      const record: EventRecord = {
        id: entry.id,
        date: entry.date,
        dayOfWeek: entry.dayOfWeek,
        time: entry.time,
        location: entry.location,
        title: entry.title,
        details: entry.details,
        category: entry.category,
        keywords: entry.keywords || [],
      };
      this.events.set(record.id, record);
      this.order.push(record.id);
    }
  }

  listEvents(): EventRecord[] {
    return this.order.map((id) => this.events.get(id)!);
  }

  getEvent(eventId: string): EventRecord | null {
    return this.events.get(eventId) || null;
  }

  searchByDate(value: string): EventRecord[] {
    const target = this.parseDate(value);
    if (!target) return [];
    return this.order
      .map((id) => this.events.get(id)!)
      .filter((r) => r.date === target);
  }

  searchByDayOfWeek(day: string): EventRecord[] {
    const normalized = day.trim().toLowerCase();
    if (!normalized) return [];
    return this.order
      .map((id) => this.events.get(id)!)
      .filter((r) => r.dayOfWeek.trim().toLowerCase() === normalized);
  }

  searchByTime(value: string): EventRecord[] {
    const target = this.parseTime(value);
    if (!target) return [];
    return this.order
      .map((id) => this.events.get(id)!)
      .filter((r) => r.time === target);
  }

  searchByKeyword(terms: string | string[]): EventRecord[] {
    const normalizedTerms = this.normalizeKeywords(terms);
    if (normalizedTerms.length === 0) return [];

    const matches: EventRecord[] = [];
    for (const id of this.order) {
      const record = this.events.get(id)!;
      const haystack = this.eventFields(record).map((f) => f.toLowerCase());
      const hasMatch = normalizedTerms.some((term) => haystack.some((f) => f.includes(term)));
      if (hasMatch) matches.push(record);
    }
    return matches;
  }

  listAvailableKeywords(): string[] {
    const keywords: Map<string, null> = new Map();
    for (const id of this.order) {
      const record = this.events.get(id)!;
      for (const keyword of record.keywords) {
        const text = keyword.trim();
        if (text) keywords.set(text.toLowerCase(), null);
      }
      const category = record.category.trim();
      if (category) keywords.set(category.toLowerCase(), null);
    }
    return [...keywords.keys()].sort();
  }

  private eventFields(record: EventRecord): string[] {
    return [
      record.id,
      record.dayOfWeek,
      record.location,
      record.title,
      record.details,
      record.category,
      record.keywords.join(" "),
    ];
  }

  private parseDate(value: string): string | null {
    const text = value.trim();
    if (!text) return null;
    try {
      const parsed = new Date(text);
      return parsed.toISOString().split("T")[0];
    } catch {
      return null;
    }
  }

  private parseTime(value: string): string | null {
    const text = value.trim();
    if (!text) return null;
    const match = text.match(/^(\d{1,2}):(\d{2})$/);
    if (!match) return null;
    return `${match[1].padStart(2, "0")}:${match[2]}`;
  }

  private normalizeKeywords(terms: string | string[]): string[] {
    const values = typeof terms === "string" ? [terms] : terms;
    const normalized: string[] = [];
    for (const value of values) {
      const text = value.trim().toLowerCase();
      if (text) {
        normalized.push(text);
        const tokens = text.split(/[^a-z0-9]+/).filter(Boolean);
        normalized.push(...tokens);
      }
    }
    return [...new Set(normalized)];
  }
}

