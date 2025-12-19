/**
 * Event list widget builder for the news guide demo.
 */

import * as path from "node:path";
import { WidgetTemplate } from "chatkit-server";
import type { EventRecord } from "../data/event_store.ts";

const widgetPath = path.resolve(Deno.cwd(), "apps/news-guide/backend-ts/widgets/event_list.widget");
const eventListWidgetTemplate = WidgetTemplate.fromFile(widgetPath);

const CATEGORY_COLORS: Record<string, string> = {
  community: "purple-400",
  civics: "blue-400",
  arts: "pink-400",
  outdoors: "green-400",
  music: "orange-400",
  family: "yellow-400",
  food: "red-400",
  fitness: "teal-400",
};
const DEFAULT_CATEGORY_COLOR = "gray-400";

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  return `${weekdays[d.getDay()]}, ${months[d.getMonth()]} ${d.getDate()}`;
}

function formatTime(record: EventRecord): string {
  const [hours, minutes] = record.time.split(":").map(Number);
  const period = hours >= 12 ? "PM" : "AM";
  const hour12 = hours % 12 || 12;
  return `${hour12}:${minutes.toString().padStart(2, "0")} ${period}`;
}

function serializeEvent(record: EventRecord): Record<string, unknown> {
  const category = (record.category || "").trim().toLowerCase();
  const color = CATEGORY_COLORS[category] || DEFAULT_CATEGORY_COLOR;
  return {
    id: record.id,
    title: record.title,
    location: record.location,
    timeLabel: formatTime(record),
    dateLabel: formatDate(record.date),
    color,
    details: record.details,
  };
}

export function buildEventListWidget(
  events: EventRecord[],
  selectedEventId: string | null = null
): unknown {
  const records = [...events].sort((a, b) => a.date.localeCompare(b.date));
  const eventIds = records.map((r) => r.id);

  // Group events by date
  const groupsMap: Map<string, { dateLabel: string; events: Record<string, unknown>[] }> = new Map();
  for (const record of records) {
    const dateLabel = formatDate(record.date);
    if (!groupsMap.has(record.date)) {
      groupsMap.set(record.date, { dateLabel, events: [] });
    }
    groupsMap.get(record.date)!.events.push(serializeEvent(record));
  }

  const groups = [...groupsMap.values()];
  const payload = {
    groups,
    selectedEventId,
    eventIds,
  };

  return eventListWidgetTemplate.build({ data: payload });
}

