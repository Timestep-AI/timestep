/**
 * Defines a widget for the cat's profile card.
 * This is an example of a static presentation widget.
 */

import { WidgetTemplate } from "chatkit-server";
import type { WidgetRoot } from "chatkit-server";
import type { CatState } from "../cat_state.ts";
import { dirname, fromFileUrl, join } from "https://deno.land/std@0.224.0/path/mod.ts";

// Get absolute path to the widget file
const currentDir = dirname(fromFileUrl(import.meta.url));
const catProfileWidgetTemplate = WidgetTemplate.fromFile(join(currentDir, "cat_profile.widget"));

function formatAgeLabel(age: number): string {
  return age === 1 ? "1 year" : `${age} years`;
}

function formatColorPatternLabel(colorPattern: string | null): string {
  if (colorPattern === null) {
    return "N/A";
  }
  return colorPattern.charAt(0).toUpperCase() + colorPattern.slice(1);
}

function formatFavoriteToy(favoriteToy: string | null): string {
  if (favoriteToy === null) {
    return "A laser pointer";
  }
  return favoriteToy.charAt(0).toUpperCase() + favoriteToy.slice(1);
}

function imageSrc(state: CatState): string {
  if (state.colorPattern === "black") {
    return "https://files.catbox.moe/pbkakb.png";
  }
  if (state.colorPattern === "calico") {
    return "https://files.catbox.moe/p2mj4g.png";
  }
  if (state.colorPattern === "colorpoint") {
    return "https://files.catbox.moe/nrtexw.png";
  }
  if (state.colorPattern === "tabby") {
    return "https://files.catbox.moe/zn77bd.png";
  }
  if (state.colorPattern === "white") {
    return "https://files.catbox.moe/zvkhpo.png";
  }
  return "https://files.catbox.moe/e42tgh.png";
}

export function buildProfileCardWidget(
  state: CatState,
  favoriteToy?: string | null
): WidgetRoot {
  return catProfileWidgetTemplate.build({
    name: state.name,
    image_src: imageSrc(state),
    age: formatAgeLabel(state.age),
    color_pattern: formatColorPatternLabel(state.colorPattern),
    favorite_toy: formatFavoriteToy(favoriteToy ?? null),
  });
}

export function profileWidgetCopyText(state: CatState): string {
  return `${state.name}, age ${state.age}, is a ${state.colorPattern} cat.`;
}

