/**
 * News Guide Server - Deno HTTP Server
 */

import { StreamingResult } from "chatkit-server";
import { NewsAssistantServer } from "./server.ts";

const newsServer = new NewsAssistantServer();

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const pathname = url.pathname;

  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, article-id",
  };

  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  // Health check
  if (pathname === "/health" && req.method === "GET") {
    return new Response(JSON.stringify({ status: "healthy" }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  // Articles metadata
  if (pathname === "/articles" && req.method === "GET") {
    const articles = newsServer.articleStore.listMetadata();
    return new Response(JSON.stringify({ articles }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  // Single article
  if (pathname.startsWith("/articles/") && req.method === "GET") {
    const articleId = pathname.slice("/articles/".length);
    const article = newsServer.articleStore.getArticle(articleId);
    if (!article) {
      return new Response(JSON.stringify({ error: "Article not found" }), {
        status: 404,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ article }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  // Events
  if (pathname === "/events" && req.method === "GET") {
    const events = newsServer.eventStore.listEvents();
    return new Response(JSON.stringify({ events }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  // Authors
  if (pathname === "/authors" && req.method === "GET") {
    const authors = newsServer.articleStore.listAuthors();
    return new Response(JSON.stringify({ authors }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  // ChatKit endpoint
  if (pathname === "/chatkit" && req.method === "POST") {
    const payload = await req.text();
    const articleId = req.headers.get("article-id") || undefined;
    const context = { request: req, article_id: articleId };
    const result = await newsServer.process(payload, context);

    if (result instanceof StreamingResult) {
      const stream = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of result) {
              controller.enqueue(new TextEncoder().encode(chunk));
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
        headers: {
          ...corsHeaders,
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    const jsonResult = result as { json?: string } | Record<string, unknown>;
    if ("json" in jsonResult && typeof jsonResult.json === "string") {
      return new Response(jsonResult.json, {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  return new Response(JSON.stringify({ error: "Not found" }), {
    status: 404,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

const port = parseInt(Deno.env.get("PORT") || "8000");
console.log(`News Guide Server starting on port ${port}...`);
Deno.serve({ port }, handler);

