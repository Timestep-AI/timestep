/**
 * Deno HTTP server entrypoint for the Cozy Cat Lounge backend.
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
import { CatAssistantServer } from "./server.ts";

const PORT = parseInt(Deno.env.get("PORT") || "8000", 10);

const chatKitServer = new CatAssistantServer();

const app = new Hono();

// CORS middleware
app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowHeaders: ["*"],
    credentials: true,
  })
);

// Health check endpoint
app.get("/", (c) => {
  return c.json({ status: "ok" });
});

// ChatKit endpoint
app.post("/chatkit", async (c) => {
  try {
    const payload = await c.req.arrayBuffer();
    const context = { request: c.req.raw };
    const result = await chatKitServer.process(new Uint8Array(payload), context);

    if (result instanceof StreamingResult) {
      // Create a readable stream from the async generator
      const stream = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of result) {
              try {
                controller.enqueue(chunk);
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

    // Non-streaming result
    if ("json" in result) {
      return c.text(result.json, 200, {
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

console.log(`Cozy Cat Lounge Backend running on http://localhost:${PORT}`);
Deno.serve({ port: PORT }, app.fetch);
