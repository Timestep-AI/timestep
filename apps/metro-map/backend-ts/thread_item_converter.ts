/**
 * Custom thread item converter for metro map.
 */

import { ThreadItemConverter } from "chatkit-server";
import type { HiddenContextItem, UserMessageTagContent } from "chatkit-server";
import type { MetroMapStore } from "./data/metro_map_store.ts";

export class MetroMapThreadItemConverter extends ThreadItemConverter {
  private metroMapStore: MetroMapStore;

  constructor(metroMapStore: MetroMapStore) {
    super();
    this.metroMapStore = metroMapStore;
  }

  async hiddenContextToInput(item: HiddenContextItem): Promise<unknown> {
    return {
      type: "message",
      role: "user",
      content: [{
        type: "input_text",
        text: typeof item.content === "string" ? item.content : JSON.stringify(item.content),
      }],
    };
  }

  async tagToMessageContent(tag: UserMessageTagContent): Promise<unknown> {
    const tagData = tag.data || {};
    const stationId = ((tagData as Record<string, unknown>).station_id as string || tag.id || "").trim();
    const stationName = ((tagData as Record<string, unknown>).name as string || tag.text || stationId).trim();

    const station = stationId ? this.metroMapStore.findStation(stationId) : null;
    if (!station) {
      return {
        type: "input_text",
        text: [
          "Tagged station (not found):",
          "<STATION_TAG>",
          `name: ${stationName}`,
          "status: not found",
          "</STATION_TAG>",
        ].join("\n"),
      };
    }

    const lineDetails: string[] = [];
    for (const lineId of station.lines) {
      const line = this.metroMapStore.findLine(lineId);
      if (line) {
        lineDetails.push(
          `- ${line.name} (id=${line.id}, color=${line.color}, orientation=${line.orientation})`
        );
      }
    }

    const text = [
      "Tagged station with full details:",
      "<STATION_TAG>",
      `id: ${station.id}`,
      `name: ${station.name}`,
      `description: ${station.description}`,
      "lines:",
      lineDetails.length > 0 ? lineDetails.join("\n") : "- none",
      "</STATION_TAG>",
    ].join("\n");

    return { type: "input_text", text };
  }
}

