/**
 * Article list widget builder for the news guide demo.
 */

import * as path from "node:path";
import { WidgetTemplate } from "chatkit-server";
import type { ArticleMetadata } from "../data/article_store.ts";

const widgetPath = path.resolve(Deno.cwd(), "apps/news-guide/backend-ts/widgets/article_list.widget");
const articleListWidgetTemplate = WidgetTemplate.fromFile(widgetPath);

function formatDate(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

function serializeArticle(article: ArticleMetadata): Record<string, unknown> {
  return {
    id: article.id,
    title: article.title,
    author: article.author,
    heroImageUrl: article.heroImageUrl,
    date: formatDate(article.date),
  };
}

export function buildArticleListWidget(articles: ArticleMetadata[]): unknown {
  const payload = { articles: articles.map(serializeArticle) };
  return articleListWidgetTemplate.build({ data: payload });
}

