/**
 * News Guide Server - Deno HTTP Server
 */

// Suppress unhandled rejections from debug package dependencies
globalThis.addEventListener("unhandledrejection", (event) => {
  if (event.reason?.message?.includes("require is not defined")) {
    event.preventDefault();
  }
});

import { Hono } from "hono";
import { cors } from "hono/cors";
import { StreamingResult } from "chatkit-server";
import { NewsAssistantServer } from "./server.ts";

const newsServer = new NewsAssistantServer();

const app = new Hono();

// CORS middleware
app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "article-id"],
  })
);

// Health check
app.get("/health", (c) => {
  return c.json({ status: "healthy" });
});

// Articles metadata
app.get("/articles", (c) => {
  const articles = newsServer.articleStore.listMetadata();
  return c.json({ articles });
});

// Single article
app.get("/articles/:article_id", (c) => {
  const articleId = c.req.param("article_id");
  const article = newsServer.articleStore.getArticle(articleId);
  if (!article) {
    return c.json({ error: "Article not found" }, 404);
  }
  return c.json({ article });
});

// Featured articles (matching FastAPI route)
app.get("/articles/featured", (c) => {
  const FEATURED_ARTICLE_IDS = [
    "unscheduled-parade-formation",
    "community-fridge-apple-pies",
    "missed-connection-reunion",
    "transit-pass-machine-updated",
  ];
  const articles = FEATURED_ARTICLE_IDS.map((id) =>
    newsServer.articleStore.getMetadata(id)
  );
  return c.json({ articles });
});

// Article tags (matching FastAPI route)
app.get("/articles/tags", (c) => {
  // This endpoint returns tags for entity search and preview requests
  // The FastAPI version builds article and author tags with preview widgets
  // TypeScript version returns basic structure without preview widgets
  function truncateTitle(value: string, maxLength: number = 30): string {
    if (value.length <= maxLength) {
      return value;
    }
    const cutoff = maxLength - 3;
    if (cutoff <= 0) {
      return "...".slice(0, maxLength);
    }
    return value.slice(0, cutoff).trimEnd() + "...";
  }

  function buildArticleTag(article: { id: string; title: string; url: string }) {
    return {
      entity: {
        title: truncateTitle(article.title),
        id: article.id,
        icon: "document",
        interactive: true,
        group: "Articles",
        data: {
          article_id: article.id,
          url: article.url,
        },
      },
      preview: null, // Preview widgets not implemented in TypeScript version
    };
  }

  function buildAuthorTag(author: { id: string; name: string; articleCount: number }) {
    return {
      entity: {
        title: author.name,
        id: `author:${author.id}`,
        icon: "profile-card",
        interactive: true,
        group: "Authors",
        data: {
          author: author.name,
          author_id: author.id,
          type: "author",
        },
      },
      preview: null, // Preview widgets not implemented in TypeScript version
    };
  }

  const articles = newsServer.articleStore.listMetadata().map(buildArticleTag);
  const authors = newsServer.articleStore.listAuthors().map(buildAuthorTag);

  return c.json({ tags: articles.concat(authors) });
});

// Events
app.get("/events", (c) => {
  const events = newsServer.eventStore.listEvents();
  return c.json({ events });
});

// Authors
app.get("/authors", (c) => {
  const authors = newsServer.articleStore.listAuthors();
  return c.json({ authors });
});

// ChatKit endpoint
app.post("/chatkit", async (c) => {
  try {
    const payload = await c.req.text();
    const articleId = c.req.header("article-id") || undefined;
    const context = { request: c.req.raw, article_id: articleId };
    const result = await newsServer.process(payload, context);

    if (result instanceof StreamingResult) {
      const stream = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of result) {
              try {
                controller.enqueue(new TextEncoder().encode(chunk));
              } catch (enqueueError) {
                // Stream may already be closed by client (e.g., during teardown)
                // This is not a real error, so we can ignore it
                return;
              }
            }
            try {
              controller.close();
            } catch (closeError) {
              // Stream may already be closed by client (e.g., during teardown)
              // This is not a real error, so we can ignore it
            }
          } catch (error) {
            console.error("Stream error:", error);
            try {
              controller.error(error);
            } catch (errorError) {
              // Stream may already be closed, ignore
            }
          }
        },
      });

      return new Response(stream, {
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    const jsonResult = result as { json?: string } | Record<string, unknown>;
    if ("json" in jsonResult && typeof jsonResult.json === "string") {
      return c.text(jsonResult.json, 200, {
        "Content-Type": "application/json",
      });
    }
    return c.json(result);
  } catch (error) {
    console.error("Error processing ChatKit request:", error);
    return c.json({ error: "Internal server error" }, 500);
  }
});

// 404 handler
app.notFound((c) => {
  return c.json({ error: "Not found" }, 404);
});

const port = parseInt(Deno.env.get("PORT") || "8000");
console.log(`News Guide Server starting on port ${port}...`);
Deno.serve({ port }, app.fetch);
