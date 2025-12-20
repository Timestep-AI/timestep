/**
 * Customer Support Server - Deno HTTP Server
 */

// Suppress unhandled rejections from debug package dependencies
globalThis.addEventListener("unhandledrejection", (event) => {
  if (event.reason?.message?.includes("require is not defined")) {
    event.preventDefault();
  }
});

import { Hono } from "hono";
import { cors } from "hono/cors";
import { run } from "@openai/agents";
import {
  ChatKitServer,
  StreamingResult,
  AgentContext,
  streamAgentResponse,
} from "chatkit-server";
import type {
  ThreadMetadata,
  ThreadStreamEvent,
  UserMessageItem,
  Action,
  WidgetItem,
  ThreadItemDoneEvent,
  HiddenContextItem,
} from "chatkit-server";
import { MemoryStore } from "./memory_store.ts";
import { AirlineStateManager, CustomerProfile } from "./airline_state.ts";
import {
  buildMealPreferenceWidget,
  mealPreferenceLabel,
  SET_MEAL_PREFERENCE_ACTION_TYPE,
  type MealPreferenceOption,
} from "./meal_preferences.ts";
import { CustomerSupportThreadItemConverter } from "./thread_item_converter.ts";
import { supportAgent, stateManager } from "./support_agent.ts";
import { titleAgent } from "./title_agent.ts";

const DEFAULT_THREAD_ID = "demo_default_thread";

function getCustomerProfileAsInputItem(profile: CustomerProfile): unknown {
  const segments: string[] = [];
  for (const segment of profile.segments) {
    segments.push(
      `- ${segment.flightNumber} ${segment.origin}->${segment.destination}` +
        ` on ${segment.date} seat ${segment.seat} (${segment.status})`
    );
  }
  const summary = segments.join("\n");
  const timeline = profile.timeline.slice(0, 3);
  const recent = timeline
    .map((entry) => `  * ${entry.entry} (${entry.timestamp})`)
    .join("\n");
  const content = `<CUSTOMER_PROFILE>
Name: ${profile.name} (${profile.loyaltyStatus})
Loyalty ID: ${profile.loyaltyId}
Contact: ${profile.email}, ${profile.phone}
Checked Bags: ${profile.bagsChecked}
Meal Preference: ${profile.mealPreference || "Not set"}
Special Assistance: ${profile.specialAssistance || "None"}
Upcoming Segments:
${summary}
Recent Service Timeline:
${recent || "  * No service actions recorded yet."}
</CUSTOMER_PROFILE>`;

  return {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: content }],
  };
}

interface SupportAgentContext extends AgentContext {
  stateManager: AirlineStateManager;
}

class CustomerSupportServer extends ChatKitServer<Record<string, unknown>> {
  private agentState: AirlineStateManager;
  private agent = supportAgent;
  private titleAgentInstance = titleAgent;
  private threadItemConverter = new CustomerSupportThreadItemConverter();

  constructor(agentState: AirlineStateManager) {
    const store = new MemoryStore();
    super(store);
    this.agentState = agentState;
  }

  async *action(
    thread: ThreadMetadata,
    action: Action<string, unknown>,
    sender: WidgetItem | null,
    context: Record<string, unknown>
  ): AsyncIterator<ThreadStreamEvent> {
    if (action.type !== SET_MEAL_PREFERENCE_ACTION_TYPE) {
      return;
    }

    const payload = this.parseMealPreferencePayload(action);
    if (payload === null) {
      return;
    }

    const mealLabel = mealPreferenceLabel(payload.meal);
    this.agentState.setMeal(thread.id, mealLabel);

    if (sender !== null) {
      const widget = buildMealPreferenceWidget(payload.meal);
      yield {
        type: "thread.item.updated",
        item_id: sender.id,
        update: { type: "widget.root", widget },
      };
      yield {
        type: "thread.item.done",
        item: {
          type: "assistant_message",
          thread_id: thread.id,
          id: this.store.generateItemId("message", thread, context),
          created_at: new Date(),
          content: [
            {
              type: "output_text",
              text: `Your meal preference has been updated to "${mealLabel}".`,
            },
          ],
        },
      } as ThreadItemDoneEvent;

      const hidden: HiddenContextItem = {
        type: "hidden_context",
        id: this.store.generateItemId("message", thread, context),
        thread_id: thread.id,
        created_at: new Date(),
        content: `<WIDGET_ACTION widgetId=${sender.id}>${action.type} was performed with payload: ${payload.meal}</WIDGET_ACTION>`,
      };
      await this.store.addThreadItem(thread.id, hidden, context);
    }
  }

  async *respond(
    thread: ThreadMetadata,
    inputUserMessage: UserMessageItem | null,
    context: Record<string, unknown>
  ): AsyncIterator<ThreadStreamEvent> {
    // Load all items from the thread to send as agent input.
    const itemsPage = await this.store.loadThreadItems(thread.id, null, 20, "desc", context);
    const updatingThreadTitle = this.maybeUpdateThreadTitle(thread, inputUserMessage);
    const items = itemsPage.data.reverse();

    // Prepend customer profile as part of the agent input
    const profile = this.agentState.getProfile(thread.id);
    const profileItem = getCustomerProfileAsInputItem(profile);
    const inputItems = [
      profileItem,
      ...(await this.threadItemConverter.toAgentInput(items)),
    ];

    const agentContext: SupportAgentContext = new AgentContext({
      thread,
      store: this.store,
      requestContext: context,
    }) as SupportAgentContext;
    agentContext.stateManager = this.agentState;

    const result = await run(this.agent, inputItems, {
      stream: true,
      context: agentContext,
      runConfig: {
        modelSettings: {
          temperature: 0.4,
        },
      },
    });

    for await (const event of streamAgentResponse(agentContext, result)) {
      yield event;
    }

    await updatingThreadTitle;
  }

  private async maybeUpdateThreadTitle(
    thread: ThreadMetadata,
    userMessage: UserMessageItem | null
  ): Promise<void> {
    if (userMessage === null || thread.title !== null) {
      return;
    }

    const runResult = await run(
      this.titleAgentInstance,
      await this.threadItemConverter.toAgentInput(userMessage)
    );
    let modelResult: string = runResult.finalOutput;
    // Capitalize the first letter only
    modelResult = modelResult.charAt(0).toUpperCase() + modelResult.slice(1);
    thread.title = modelResult.replace(/\.$/, "");
  }

  private parseMealPreferencePayload(
    action: Action<string, unknown>
  ): { meal: MealPreferenceOption } | null {
    const payload = action.payload as { meal?: string } | null | undefined;
    if (!payload || typeof payload.meal !== "string") {
      console.warn("Invalid meal preference payload:", action.payload);
      return null;
    }
    const validMeals = ["vegetarian", "kosher", "gluten intolerant", "child"];
    if (!validMeals.includes(payload.meal)) {
      console.warn("Invalid meal preference value:", payload.meal);
      return null;
    }
    return { meal: payload.meal as MealPreferenceOption };
  }
}

// Create server instance
const supportServer = new CustomerSupportServer(stateManager);

// Hono app
const app = new Hono();

// CORS middleware
app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type"],
  })
);

// Health check
app.get("/support/health", (c) => {
  return c.json({ status: "healthy" });
});

// Customer snapshot
app.get("/support/customer", (c) => {
  const threadId = c.req.query("thread_id") || DEFAULT_THREAD_ID;
  const data = stateManager.toDict(threadId);
  return c.json({ customer: data });
});

// ChatKit endpoint
app.post("/support/chatkit", async (c) => {
  try {
    const payload = await c.req.text();
    const result = await supportServer.process(payload, { request: c.req.raw });

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

    // Non-streaming response
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

// Start server
const port = parseInt(Deno.env.get("PORT") || "8000");
console.log(`Customer Support Server starting on port ${port}...`);
Deno.serve({ port }, app.fetch);
