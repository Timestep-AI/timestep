/**
 * Airline customer support agent with task-specific tools.
 */

import { Agent, tool } from "@openai/agents";
import { z } from "zod";
import type { AgentContext } from "chatkit-server";
import type { ThreadItemDoneEvent } from "chatkit-server";
import { AirlineStateManager } from "./airline_state.ts";
import { buildMealPreferenceWidget } from "./meal_preferences.ts";

const SUPPORT_AGENT_INSTRUCTIONS = `
You are a friendly and efficient airline customer support agent for OpenSkies.
You help elite flyers with seat changes, cancellations, checked bags, and
special requests. Follow these guidelines:

- Acknowledge the customer's loyalty status and recent travel plans if you haven't
  already done so.
- When a task requires action, call the appropriate tool instead of describing
  the change hypothetically.
- After using a tool, confirm the outcome and offer next steps.
- If you cannot fulfill a request, apologize and suggest an alternative.
- Keep responses concise (2-3 sentences) unless extra detail is required.
- For tool calls \`cancel_trip\` and \`add_checked_bag\`, ask the user for confirmation before proceeding.

Custom tags:
- <CUSTOMER_PROFILE> - provides contexto on the customer's account and travel details.

Available tools:
- change_seat(flight_number: str, seat: str) – move the passenger to a new seat.
- cancel_trip() – cancel the upcoming reservation and note the refund.
- add_checked_bag() – add one checked bag to the itinerary.
- meal_preference_list() – show meal options so the traveller can pick their preference.
  Invoke this tool when the user requests to set or change their meal preference or option.
- request_assistance(note: str) – record a special assistance request.

Only use information provided in the customer context or tool results. Do not
invent confirmation numbers or policy details.
`.trim();

// AgentContext with state manager access
export interface SupportAgentContext extends AgentContext {
  stateManager: AirlineStateManager;
}

// Helper to extract context from SDK wrapper
function getContext(ctx: unknown): SupportAgentContext {
  const wrapper = ctx as { context: SupportAgentContext };
  return wrapper.context || (ctx as SupportAgentContext);
}

function getThreadId(ctx: unknown): string {
  return getContext(ctx).thread.id;
}

export function buildSupportAgent(stateManager: AirlineStateManager): Agent {
  // Tool: Change seat
  const changeSeatTool = tool({
    name: "change_seat",
    description: "Move the passenger to a different seat on a flight.",
    parameters: z.object({
      flight_number: z.string().describe("Flight number to change seat on"),
      seat: z.string().describe("New seat assignment (e.g., 12C)"),
    }),
    execute: async (args, ctx) => {
      const threadId = getThreadId(ctx);
      try {
        const message = stateManager.changeSeat(threadId, args.flight_number, args.seat);
        return { result: message };
      } catch (error) {
        throw new Error(error instanceof Error ? error.message : String(error));
      }
    },
  });

  // Tool: Cancel trip
  const cancelTripTool = tool({
    name: "cancel_trip",
    description: "Cancel the traveller's upcoming trip and note the refund.",
    parameters: z.object({}),
    execute: async (_args, ctx) => {
      const threadId = getThreadId(ctx);
      const message = stateManager.cancelTrip(threadId);
      return { result: message };
    },
  });

  // Tool: Add checked bag
  const addCheckedBagTool = tool({
    name: "add_checked_bag",
    description: "Add a checked bag to the reservation.",
    parameters: z.object({}),
    execute: async (_args, ctx) => {
      const threadId = getThreadId(ctx);
      const message = stateManager.addBag(threadId);
      const profile = stateManager.getProfile(threadId);
      return { result: message, bags_checked: profile.bagsChecked };
    },
  });

  // Tool: Meal preference list
  const mealPreferenceListTool = tool({
    name: "meal_preference_list",
    description: "Display the meal preference picker so the traveller can choose an option.",
    parameters: z.object({}),
    execute: async (_args, ctx) => {
      const agentCtx = getContext(ctx);

      // Send message first
      await agentCtx.stream({
        type: "thread.item.done",
        item: {
          type: "assistant_message",
          thread_id: agentCtx.thread.id,
          id: agentCtx.generateId("message"),
          created_at: new Date(),
          content: [{ type: "output_text", text: "Please select your meal preference." }],
        },
      } as ThreadItemDoneEvent);

      // Then send widget
      const widget = buildMealPreferenceWidget();
      await agentCtx.streamWidget(widget);

      return { result: "Shared meal preference options with the traveller." };
    },
  });

  // Tool: Request assistance
  const requestAssistanceTool = tool({
    name: "request_assistance",
    description: "Note a special assistance request for airport staff.",
    parameters: z.object({
      note: z.string().describe("Special assistance request details"),
    }),
    execute: async (args, ctx) => {
      const threadId = getThreadId(ctx);
      const message = stateManager.requestAssistance(threadId, args.note);
      return { result: message };
    },
  });

  return new Agent({
    model: "gpt-4.1-mini",
    name: "OpenSkies Concierge",
    instructions: SUPPORT_AGENT_INSTRUCTIONS,
    tools: [
      changeSeatTool,
      cancelTripTool,
      addCheckedBagTool,
      mealPreferenceListTool,
      requestAssistanceTool,
    ],
    // Stop inference after tool calls with widget outputs to prevent repetition.
    toolUseBehavior: {
      stopAtToolNames: [mealPreferenceListTool.name],
    },
  });
}

// Singleton instances
export const stateManager = new AirlineStateManager();
export const supportAgent = buildSupportAgent(stateManager);

