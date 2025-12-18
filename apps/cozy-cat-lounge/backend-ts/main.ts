/**
 * Deno HTTP server entrypoint for the Cozy Cat Lounge backend.
 */

import { StreamingResult } from "chatkit-server";
import { CatAssistantServer } from "./server.ts";

const PORT = parseInt(Deno.env.get("PORT") || "8000", 10);

const chatKitServer = new CatAssistantServer();

// CORS headers
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Allow-Credentials": "true",
};

async function handler(request: Request): Promise<Response> {
  const url = new URL(request.url);

  // Handle CORS preflight
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: corsHeaders,
    });
  }

  // Health check endpoint
  if (url.pathname === "/" && request.method === "GET") {
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json",
      },
    });
  }

  // ChatKit endpoint
  if (url.pathname === "/chatkit" && request.method === "POST") {
    try {
      const payload = await request.arrayBuffer();
      const context = { request };
      const result = await chatKitServer.process(new Uint8Array(payload), context);

      if (result instanceof StreamingResult) {
        // Create a readable stream from the async generator
        const stream = new ReadableStream({
          async start(controller) {
            try {
              for await (const chunk of result) {
                controller.enqueue(chunk);
              }
              controller.close();
            } catch (error) {
              controller.error(error);
            }
          },
        });

        return new Response(stream, {
          status: 200,
          headers: {
            ...corsHeaders,
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
          },
        });
      }

      // Non-streaming result
      if ("json" in result) {
        return new Response(result.json, {
          status: 200,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        });
      }

      return new Response(JSON.stringify(result), {
        status: 200,
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      });
    } catch (error) {
      console.error("Error processing ChatKit request:", error);
      return new Response(
        JSON.stringify({ error: "Internal server error" }),
        {
          status: 500,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        }
      );
    }
  }

  // Not found
  return new Response(JSON.stringify({ error: "Not found" }), {
    status: 404,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json",
    },
  });
}

console.log(`Cozy Cat Lounge Backend running on http://localhost:${PORT}`);
Deno.serve({ port: PORT }, handler);

