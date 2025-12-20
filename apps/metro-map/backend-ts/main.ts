/**
 * Metro Map Server - Deno HTTP Server
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
import { MetroMapServer } from "./server.ts";
import type { MetroMap } from "./data/metro_map_store.ts";

const metroMapServer = new MetroMapServer();

const app = new Hono();

// CORS middleware
app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "map-id"],
  })
);

// Health check
app.get("/health", (c) => {
  return c.json({ status: "healthy" });
});

// Get map
app.get("/map", (c) => {
  const map = metroMapServer.metroMapStore.dumpForClient();
  return c.json({ map });
});

// Update map
app.post("/map", async (c) => {
  try {
    const body = await c.req.json();
    const map = body.map as MetroMap;
    metroMapServer.metroMapStore.setMap(map);
    return c.json({ map });
  } catch (error) {
    return c.json(
      { error: error instanceof Error ? error.message : "Invalid request" },
      400
    );
  }
});

// ChatKit endpoint
app.post("/chatkit", async (c) => {
  try {
    const payload = await c.req.text();
    const mapId = c.req.header("map-id") || "solstice-metro";
    const context = { request: c.req.raw, map_id: mapId };
    const result = await metroMapServer.process(payload, context);

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
console.log(`Metro Map Server starting on port ${port}...`);
Deno.serve({ port }, app.fetch);
