/**
 * Defines an interactive widget that prompts the user to select a name for the cat.
 * This widget wires up two client action configs.
 */

import { WidgetTemplate } from "chatkit-server";
import type { WidgetRoot } from "chatkit-server";
import { dirname, fromFileUrl, join } from "https://deno.land/std@0.224.0/path/mod.ts";

export interface CatNameSuggestion {
  name: string;
  reason?: string | null;
}

export interface CatNameSelectionPayload {
  name: string;
  options: CatNameSuggestion[];
}

// Get absolute path to the widget file
const currentDir = dirname(fromFileUrl(import.meta.url));
const nameSuggestionsWidgetTemplate = WidgetTemplate.fromFile(join(currentDir, "cat_name_suggestions.widget"));

export function buildNameSuggestionsWidget(
  names: CatNameSuggestion[],
  selected?: string | null
): WidgetRoot {
  console.log(`Building name suggestions widget with selected: ${selected}`);
  console.log(`Names: ${JSON.stringify(names)}`);
  
  const titleCase = (str: string) =>
    str.split(" ").map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(" ");

  return nameSuggestionsWidgetTemplate.build({
    items: names.map((suggestion) => ({
      name: suggestion.name,
      reason: suggestion.reason,
    })),
    selected: selected ? titleCase(selected.trim()) : null,
  });
}

