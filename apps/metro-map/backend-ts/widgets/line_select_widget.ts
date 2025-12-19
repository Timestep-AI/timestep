/**
 * Line select widget builder for the metro map demo.
 */

import * as path from "node:path";
import { WidgetTemplate } from "chatkit-server";
import type { Line } from "../data/metro_map_store.ts";

// Load widget template from file
const widgetPath = path.resolve(Deno.cwd(), "apps/metro-map/backend-ts/widgets/line_select.widget");
const lineSelectWidgetTemplate = WidgetTemplate.fromFile(widgetPath);

export function buildLineSelectWidget(
  lines: Line[],
  selected: string | null = null
): unknown {
  return lineSelectWidgetTemplate.build({
    data: {
      items: lines,
      selected,
    },
  });
}

