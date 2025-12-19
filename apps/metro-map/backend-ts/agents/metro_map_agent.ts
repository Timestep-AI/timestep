/**
 * Metro map planning agent with tools for map manipulation.
 */

import { Agent, tool } from "@openai/agents";
import { z } from "zod";
import type { AgentContext, ThreadItemDoneEvent } from "chatkit-server";
import type { MetroMapStore, Station, Line, MetroMap } from "../data/metro_map_store.ts";
import { buildLineSelectWidget } from "../widgets/line_select_widget.ts";

const INSTRUCTIONS = `
    You are a concise metro planner helping city planners update the Orbital Transit map.
    Give short answers, list 2–3 options, and highlight the lines or interchanges involved.

    Before recommending a route, sync the latest map with the provided tools. Cite line
    colors when helpful (e.g., "take Red then Blue at Central Exchange").

    When the user asks what to do next, reply with 2 concise follow-up ideas and pick one to lead with.
    Default to actionable options like adding another station on the same line or explaining how to travel
    from the newly added station to a nearby destination.

    When the user mentions a station, always call the \`get_map\` tool to sync the latest map before responding.

    When a user wants to add a station (e.g. "I would like to add a new metro station." or "Add another station"):
    - If the user did not specify a line, you MUST call \`show_line_selector\` with a message prompting them to choose one
      from the list of lines. You must NEVER ask the user to choose a line without calling \`show_line_selector\` first.
      This applies even if you just added a station—treat each new "add a station" turn as needing a fresh line selection
      unless the user explicitly included the line in that same turn or in the latest message via <LINE_SELECTED>.
    - If the user replies with a number to pick one of your follow-up options AND that option involves adding a station,
      treat this as a fresh station-add request and immediately call \`show_line_selector\` before asking anything else.
    - If the user did not specify a station name, ask them to enter a name.
    - If the user did not specify whether to add the station to the end of the line or the beginning, ask them to choose one.
    - When you have all the information you need, call the \`add_station\` tool with the station name, line id, and append flag.

    Describing:
    - After a new station has been added, describe it to the user in a whimsical and poetic sentence.
    - When describing a station to the user, omit the station id and coordinates.
    - When describing a line to the user, omit the line id and color.

    When a user wants to plan a route:
    - If the user did not specify a starting or detination station, ask them to choose them from the list of stations.
    - You MUST call the \`plan_route\` tool with the list of stations in the route and a one-sentence message describing the route.
    - The message describing the route should include the estimated travel time in light years (e.g. "10.6 light years"),
      and points of interest along the way.
    - Avoid over-explaining and stay within the given station list.

    Every time your response mentions a list of stations (e.g. "the stations on the Blue Line are..." or "to get from Titan Border to
    Lyra Verge..."), you MUST call the \`cite_stations_for_route\` tool with the list of stations.

    Custom tags:
    - <LINE_SELECTED>{line_id}</LINE_SELECTED> - when the user has selected a line, you can use this tag to reference the line id.
      When this is the latest message, acknowledge the selection.
    - <STATION_TAG>...</STATION_TAG> - contains full station details (id, name, description, coordinates, and served lines with ids/colors/orientations).
      Use the data inside the tag directly; do not call \`get_station\` just to resolve a tagged station.

    When the user mentions "selected stations" or asks about the current selection, call \`get_selected_stations\` to fetch the station ids from the client.
`.trim();

// Agent context with metro store access
export interface MetroAgentContext extends AgentContext {
  metro: MetroMapStore;
}

// Helper to extract context from SDK wrapper
function getContext(ctx: unknown): MetroAgentContext {
  const wrapper = ctx as { context: MetroAgentContext };
  return wrapper.context || (ctx as MetroAgentContext);
}

// Station schema for tool parameters
const StationSchema = z.object({
  id: z.string(),
  name: z.string(),
  x: z.number(),
  y: z.number(),
  description: z.string(),
  lines: z.array(z.string()),
});

// Tool: Show line selector widget
const showLineSelectorTool = tool({
  name: "show_line_selector",
  description: "Show a clickable widget listing metro lines.\n- `message`: Text shown above the widget to prompt the user.",
  parameters: z.object({
    message: z.string().describe("Text shown above the widget"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const widget = buildLineSelectWidget(agentCtx.metro.listLines());
    
    await agentCtx.stream({
      type: "thread.item.done",
      item: {
        type: "assistant_message",
        thread_id: agentCtx.thread.id,
        id: agentCtx.generateId("message"),
        created_at: new Date(),
        content: [{ type: "output_text", text: args.message }],
      },
    } as ThreadItemDoneEvent);
    
    await agentCtx.streamWidget(widget);
    return { success: true };
  },
});

// Tool: Get map
const getMapTool = tool({
  name: "get_map",
  description: "Load the latest metro map with lines and stations. No parameters.",
  parameters: z.object({}),
  execute: async (_args, ctx) => {
    const agentCtx = getContext(ctx);
    await agentCtx.stream({ type: "progress.update", text: "Retrieving the latest metro map..." });
    return { map: agentCtx.metro.getMap() };
  },
});

// Tool: List lines
const listLinesTool = tool({
  name: "list_lines",
  description: "List all metro lines with their colors and endpoints. No parameters.",
  parameters: z.object({}),
  execute: async (_args, ctx) => {
    const agentCtx = getContext(ctx);
    return { lines: agentCtx.metro.listLines() };
  },
});

// Tool: List stations
const listStationsTool = tool({
  name: "list_stations",
  description: "List all stations and which lines serve them. No parameters.",
  parameters: z.object({}),
  execute: async (_args, ctx) => {
    const agentCtx = getContext(ctx);
    return { stations: agentCtx.metro.listStations() };
  },
});

// Tool: Plan route
const planRouteTool = tool({
  name: "plan_route",
  description: "Show the user the planned route.\n- `route`: Ordered list of stations in the journey.\n- `message`: One-sentence description of the itinerary.",
  parameters: z.object({
    route: z.array(StationSchema).describe("Ordered list of stations"),
    message: z.string().describe("Description of the itinerary"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    
    const sources = args.route.map((station) => ({
      id: station.id,
      icon: "map-pin",
      title: station.name,
      description: station.description,
      data: { type: "station", station_id: station.id, name: station.name },
    }));
    
    await agentCtx.stream({
      type: "thread.item.done",
      item: {
        type: "assistant_message",
        thread_id: agentCtx.thread.id,
        id: agentCtx.generateId("message"),
        created_at: new Date(),
        content: [{
          type: "output_text",
          text: args.message,
          annotations: sources.map((source) => ({ source, index: 0 })),
        }],
      },
    } as ThreadItemDoneEvent);
    
    return { success: true };
  },
});

// Tool: Get station
const getStationTool = tool({
  name: "get_station",
  description: "Look up a single station and the lines serving it.\n- `station_id`: Station identifier to retrieve.",
  parameters: z.object({
    station_id: z.string().describe("Station identifier"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const station = agentCtx.metro.findStation(args.station_id);
    if (!station) {
      throw new Error(`Station '${args.station_id}' was not found.`);
    }
    const lines = station.lines
      .map((lineId) => agentCtx.metro.findLine(lineId))
      .filter((l): l is Line => l !== null);
    return { station, lines };
  },
});

// Tool: Add station
const addStationTool = tool({
  name: "add_station",
  description: "Add a new station to the metro map.\n- `station_name`: The name of the station to add.\n- `line_id`: The id of the line to add the station to.\n- `append`: Whether to add the station to the end of the line or the beginning. Defaults to True.",
  parameters: z.object({
    station_name: z.string().describe("Name of the station"),
    line_id: z.string().describe("Line ID to add station to"),
    append: z.boolean().optional().describe("Add to end (true) or beginning (false)"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const stationName = args.station_name.trim().split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(" ");
    
    await agentCtx.stream({ type: "progress.update", text: "Adding station..." });
    
    try {
      const [updatedMap, newStation] = agentCtx.metro.addStation(
        stationName,
        args.line_id,
        args.append !== false
      );
      
      await agentCtx.stream({
        type: "client.effect",
        name: "add_station",
        data: {
          stationId: newStation.id,
          map: updatedMap,
        },
      });
      
      return { map: updatedMap };
    } catch (error) {
      await agentCtx.stream({
        type: "thread.item.done",
        item: {
          type: "assistant_message",
          thread_id: agentCtx.thread.id,
          id: agentCtx.generateId("message"),
          created_at: new Date(),
          content: [{ type: "output_text", text: `There was an error adding **${stationName}**` }],
        },
      } as ThreadItemDoneEvent);
      throw error;
    }
  },
});

// Tool: Get selected stations (client tool)
const getSelectedStationsTool = tool({
  name: "get_selected_stations",
  description: "Fetch the ids of the currently selected stations from the client UI. No parameters.",
  parameters: z.object({}),
  execute: async (_args, ctx) => {
    const agentCtx = getContext(ctx);
    await agentCtx.stream({ type: "progress.update", text: "Fetching selected stations from the map..." });
    agentCtx.clientToolCall = {
      name: "get_selected_stations",
      arguments: {},
    };
    return { station_ids: [] };
  },
});

// Create the metro map agent
export const metroMapAgent = new Agent({
  model: "gpt-4o-mini",
  name: "metro_map",
  instructions: INSTRUCTIONS,
  tools: [
    getMapTool,
    listLinesTool,
    listStationsTool,
    getStationTool,
    planRouteTool,
    showLineSelectorTool,
    addStationTool,
    getSelectedStationsTool,
  ],
  // Stop inference after client tool call or widget output
  toolUseBehavior: {
    stopAtToolNames: [
      planRouteTool.name,
      showLineSelectorTool.name,
      getSelectedStationsTool.name,
    ],
  },
});

