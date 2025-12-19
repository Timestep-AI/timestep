/**
 * Metro Map Server - Deno HTTP Server
 */

import { StreamingResult } from "chatkit-server";
import { MetroMapServer } from "./server.ts";
import type { MetroMap } from "./data/metro_map_store.ts";

const metroMapServer = new MetroMapServer();

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const path = url.pathname;

  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, map-id",
  };

  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  // Health check
  if (path === "/health" && req.method === "GET") {
    return new Response(JSON.stringify({ status: "healthy" }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  // Get map
  if (path === "/map" && req.method === "GET") {
    const map = metroMapServer.metroMapStore.dumpForClient();
    return new Response(JSON.stringify({ map }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  // Update map
  if (path === "/map" && req.method === "POST") {
    try {
      const body = await req.json();
      const map = body.map as MetroMap;
      metroMapServer.metroMapStore.setMap(map);
      return new Response(JSON.stringify({ map }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    } catch (error) {
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Invalid request" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
  }

  // ChatKit endpoint
  if (path === "/chatkit" && req.method === "POST") {
    const payload = await req.text();
    const mapId = req.headers.get("map-id") || "solstice-metro";
    const context = { request: req, map_id: mapId };
    const result = await metroMapServer.process(payload, context);

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
console.log(`Metro Map Server starting on port ${port}...`);
Deno.serve({ port }, handler);

