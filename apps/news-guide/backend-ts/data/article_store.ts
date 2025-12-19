/**
 * Data models and store for News Guide demo articles.
 */

import * as path from "node:path";

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export interface ArticleMetadata {
  id: string;
  title: string;
  heroImage: string;
  heroImageUrl: string;
  url: string;
  filename: string;
  date: Date | string;
  author: string;
  tags: string[];
  keywords: string[];
}

export interface ArticleRecord extends ArticleMetadata {
  content: string;
}

export class ArticleStore {
  private dataDir: string;
  private articles: Map<string, ArticleRecord> = new Map();
  private order: string[] = [];

  constructor(dataDir: string) {
    this.dataDir = dataDir;
    this.reload();
  }

  get articlesPath(): string {
    return path.join(this.dataDir, "articles");
  }

  get metadataPath(): string {
    return path.join(this.dataDir, "articles.json");
  }

  reload(): void {
    const metadataEntries = this.loadMetadata();
    this.articles.clear();
    this.order = [];

    for (const entry of metadataEntries) {
      const markdown = this.loadMarkdown(entry);
      const record: ArticleRecord = { ...entry, content: markdown };
      this.articles.set(record.id, record);
      this.order.push(record.id);
    }
  }

  private loadMetadata(): ArticleMetadata[] {
    const raw = JSON.parse(Deno.readTextFileSync(this.metadataPath));
    if (!Array.isArray(raw)) {
      throw new Error("articles.json must contain a list of article entries.");
    }
    return raw.map((entry: Record<string, unknown>, idx: number) => {
      if (!entry.id || !entry.title) {
        throw new Error(`Invalid article metadata at index ${idx}`);
      }
      return {
        id: entry.id as string,
        title: entry.title as string,
        heroImage: (entry.heroImage || "") as string,
        heroImageUrl: (entry.heroImageUrl || "") as string,
        url: (entry.url || "") as string,
        filename: (entry.filename || "") as string,
        date: new Date(entry.date as string),
        author: (entry.author || "") as string,
        tags: (entry.tags || []) as string[],
        keywords: (entry.keywords || []) as string[],
      };
    });
  }

  private loadMarkdown(metadata: ArticleMetadata): string {
    const markdownPath = path.join(this.articlesPath, metadata.filename);
    return Deno.readTextFileSync(markdownPath);
  }

  listMetadata(): ArticleMetadata[] {
    return this.order.map((id) => this.articles.get(id)!);
  }

  listMetadataForTags(tags: string[] | null = null): ArticleMetadata[] {
    if (!tags || tags.length === 0) return this.listMetadata();
    const normalized = new Set(tags.map((t) => t.toLowerCase()));
    return this.order
      .map((id) => this.articles.get(id)!)
      .filter((record) => record.tags.some((t) => normalized.has(t.toLowerCase())));
  }

  getArticle(articleId: string): Record<string, unknown> | null {
    const record = this.articles.get(articleId);
    if (!record) return null;
    return {
      ...record,
      date: record.date instanceof Date ? record.date.toISOString() : record.date,
    };
  }

  getMetadata(articleId: string): Record<string, unknown> | null {
    const record = this.articles.get(articleId);
    if (!record) return null;
    const { content: _, ...metadata } = record;
    return {
      ...metadata,
      date: record.date instanceof Date ? record.date.toISOString() : record.date,
    };
  }

  listAuthors(): Array<{ id: string; name: string; articleCount: number }> {
    const authors: Map<string, { id: string; name: string; articleCount: number }> = new Map();
    for (const id of this.order) {
      const record = this.articles.get(id)!;
      if (!record.author) continue;
      const authorSlug = slugify(record.author);
      const entry = authors.get(authorSlug) || { id: authorSlug, name: record.author, articleCount: 0 };
      entry.articleCount++;
      authors.set(authorSlug, entry);
    }
    return [...authors.values()].sort((a, b) => a.name.localeCompare(b.name));
  }

  searchMetadataByKeywords(keywords: string[]): Record<string, unknown>[] {
    const sanitized = keywords.filter((k) => k?.trim()).map((k) => k.trim().toLowerCase());
    if (sanitized.length === 0) return [];

    const searchTerms = new Set(sanitized);
    const combined = sanitized.join(" ").trim();
    if (combined) searchTerms.add(combined);
    for (const term of [...searchTerms]) {
      const tokens = term.split(/[^a-z0-9]+/).filter(Boolean);
      tokens.forEach((t) => searchTerms.add(t));
    }

    const matches: Record<string, unknown>[] = [];
    for (const id of this.order) {
      const record = this.articles.get(id)!;
      const fields = this.metadataSearchFields(record);
      const hasMatch = [...searchTerms].some((term) => fields.some((f) => f.includes(term)));
      if (hasMatch) {
        const { content: _, ...metadata } = record;
        matches.push({
          ...metadata,
          date: record.date instanceof Date ? record.date.toISOString() : record.date,
        });
      }
    }
    return matches;
  }

  searchContentByExactText(text: string): Record<string, unknown>[] {
    const trimmed = text.trim();
    if (!trimmed) return [];

    const matches: Record<string, unknown>[] = [];
    for (const id of this.order) {
      const record = this.articles.get(id)!;
      if (record.content.includes(trimmed)) {
        const { content: _, ...metadata } = record;
        matches.push({
          ...metadata,
          date: record.date instanceof Date ? record.date.toISOString() : record.date,
        });
      }
    }
    return matches;
  }

  private metadataSearchFields(record: ArticleRecord): string[] {
    const fields = [
      record.id,
      record.title,
      record.author,
      record.url,
      record.filename,
      ...record.tags,
      ...record.keywords,
      record.date instanceof Date ? record.date.toISOString() : record.date,
    ];
    return fields.filter(Boolean).map((f) => String(f).toLowerCase());
  }

  listAvailableTagsAndKeywords(): { tags: string[]; keywords: string[] } {
    const tags = new Set<string>();
    const keywords = new Set<string>();
    for (const id of this.order) {
      const record = this.articles.get(id)!;
      record.tags.filter(Boolean).forEach((t) => tags.add(t));
      record.keywords.filter(Boolean).forEach((k) => keywords.add(k));
    }
    return {
      tags: [...tags].sort(),
      keywords: [...keywords].sort(),
    };
  }

  searchMetadataByAuthor(author: string): Record<string, unknown>[] {
    const normalized = author.trim().toLowerCase();
    if (!normalized) return [];
    const normalizedSlug = slugify(normalized);

    const matches: Record<string, unknown>[] = [];
    for (const id of this.order) {
      const record = this.articles.get(id)!;
      const authorName = record.author.toLowerCase();
      const authorSlug = slugify(record.author);
      if (
        normalized.includes(authorName) ||
        authorName.includes(normalized) ||
        normalizedSlug === authorSlug ||
        authorSlug.includes(normalizedSlug)
      ) {
        const { content: _, ...metadata } = record;
        matches.push({
          ...metadata,
          date: record.date instanceof Date ? record.date.toISOString() : record.date,
        });
      }
    }
    return matches;
  }
}

