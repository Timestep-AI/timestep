/**
 * Foxhollow Event Finder agent for discovering local events.
 */

import { Agent, tool } from "@openai/agents";
import { z } from "zod";
import type { AgentContext, ThreadItemDoneEvent } from "chatkit-server";
import type { EventStore, EventRecord } from "../data/event_store.ts";
import { buildEventListWidget } from "../widgets/event_list_widget.ts";

const INSTRUCTIONS = `
    You help Foxhollow residents discover local happenings. When a reader asks for events,
    search the curated calendar, call out dates and notable details, and keep recommendations brief.

    Use the available tools deliberately:
      - Call \`list_available_event_keywords\` to get the full set of event keywords and categories,
        fuzzy match the reader's phrasing to the closest options (case-insensitive, partial matches are ok),
        then feed those terms into a keyword search instead of relying on hard-coded synonyms.
      - If they mention a specific date (YYYY-MM-DD), start with \`search_events_by_date\`.
      - If they reference a day of the week, try \`search_events_by_day_of_week\`.
      - For general vibes (e.g., "family friendly night markets"), use \`search_events_by_keyword\`
        so the search spans titles, categories, locations, and curated keywords.

    Whenever a search tool returns more than one event immediately call \`show_event_list_widget\`
    with those results before sending your final text, along with a 1-sentence message explaining why these events were selected.
    This ensures every response ships with the timeline widget.
    Cite event titles in bold, mention the date, and highlight one delightful detail when replying.

    When the user explicitly asks for more details on the events, you MUST describe the events in natural language
    without using the \`show_event_list_widget\` tool.
`.trim();

const MODEL = "gpt-4.1-mini";

export interface EventFinderContext extends AgentContext {
  events: EventStore;
}

// Helper to extract context from SDK wrapper
function getContext(ctx: unknown): EventFinderContext {
  const wrapper = ctx as { context: EventFinderContext };
  return wrapper.context || (ctx as EventFinderContext);
}

const EventRecordSchema = z.object({
  id: z.string(),
  date: z.string(),
  dayOfWeek: z.string(),
  time: z.string(),
  location: z.string(),
  title: z.string(),
  details: z.string(),
  category: z.string(),
  keywords: z.array(z.string()),
});

// Tool: Search events by date
const searchEventsByDateTool = tool({
  name: "search_events_by_date",
  description: "Find scheduled events happening on a specific date (YYYY-MM-DD).",
  parameters: z.object({
    date: z.string().describe("Date in YYYY-MM-DD format"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    if (!args.date) throw new Error("Provide a valid date in YYYY-MM-DD format.");
    await agentCtx.stream({ type: "progress.update", text: `Looking up events on ${args.date}` });
    const records = agentCtx.events.searchByDate(args.date);
    return { events: records };
  },
});

// Tool: Search events by day of week
const searchEventsByDayOfWeekTool = tool({
  name: "search_events_by_day_of_week",
  description: "List events occurring on a given day of the week.",
  parameters: z.object({
    day: z.string().describe("Day of the week (e.g., Saturday)"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    if (!args.day) throw new Error("Provide a day of the week to search for (e.g., Saturday).");
    await agentCtx.stream({ type: "progress.update", text: `Checking ${args.day} events` });
    const records = agentCtx.events.searchByDayOfWeek(args.day);
    return { events: records };
  },
});

// Tool: Search events by keyword
const searchEventsByKeywordTool = tool({
  name: "search_events_by_keyword",
  description: "Search events with general keywords (title, category, location, or details).",
  parameters: z.object({
    keywords: z.array(z.string()).describe("Keywords to search for"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const tokens = args.keywords.filter((k) => k?.trim());
    if (tokens.length === 0) throw new Error("Provide at least one keyword to search for.");
    const label = tokens.join(", ");
    await agentCtx.stream({ type: "progress.update", text: `Searching for: ${label}` });
    const records = agentCtx.events.searchByKeyword(tokens);
    return { events: records };
  },
});

// Tool: List available event keywords
const listAvailableEventKeywordsTool = tool({
  name: "list_available_event_keywords",
  description: "List all unique event keywords and categories.",
  parameters: z.object({}),
  execute: async (_args, ctx) => {
    const agentCtx = getContext(ctx);
    await agentCtx.stream({ type: "progress.update", text: "Referencing available event keywords..." });
    return { keywords: agentCtx.events.listAvailableKeywords() };
  },
});

// Tool: Show event list widget
const showEventListWidgetTool = tool({
  name: "show_event_list_widget",
  description: "Show a timeline-styled widget for a provided set of events.",
  parameters: z.object({
    events: z.array(EventRecordSchema).describe("Events to display"),
    message: z.string().nullable().describe("Summary message"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const records = args.events.filter(Boolean);

    if (records.length === 0) {
      const fallback = args.message || "I couldn't find any events that match that search.";
      await agentCtx.stream({
        type: "thread.item.done",
        item: {
          type: "assistant_message",
          thread_id: agentCtx.thread.id,
          id: agentCtx.generateId("message"),
          created_at: new Date(),
          content: [{ type: "output_text", text: fallback }],
        },
      } as ThreadItemDoneEvent);
      return { success: true };
    }

    const widget = buildEventListWidget(records as EventRecord[]);
    const copyText = records.map((e) => e.title).filter(Boolean).join(", ");
    await agentCtx.streamWidget(widget, copyText || "Local events");

    const summary = args.message || "Here are the events that match your request.";
    await agentCtx.stream({
      type: "thread.item.done",
      item: {
        type: "assistant_message",
        thread_id: agentCtx.thread.id,
        id: agentCtx.generateId("message"),
        created_at: new Date(),
        content: [{ type: "output_text", text: summary }],
      },
    } as ThreadItemDoneEvent);

    return { success: true };
  },
});

export const eventFinderAgent = new Agent({
  model: MODEL,
  name: "Foxhollow Event Finder",
  instructions: INSTRUCTIONS,
  tools: [
    searchEventsByDateTool,
    searchEventsByDayOfWeekTool,
    searchEventsByKeywordTool,
    listAvailableEventKeywordsTool,
    showEventListWidgetTool,
  ],
  toolUseBehavior: {
    stopAtToolNames: [showEventListWidgetTool.name],
  },
});

