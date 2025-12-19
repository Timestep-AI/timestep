/**
 * Meal preference widget builder and types.
 */

export type MealPreferenceOption = "vegetarian" | "kosher" | "gluten intolerant" | "child";

export const SET_MEAL_PREFERENCE_ACTION_TYPE = "support.set_meal_preference";

export interface SetMealPreferencePayload {
  meal: MealPreferenceOption;
}

const MEAL_PREFERENCE_LABELS: Record<MealPreferenceOption, string> = {
  "vegetarian": "Vegetarian",
  "kosher": "Kosher",
  "gluten intolerant": "Gluten intolerant",
  "child": "Child",
};

export const MEAL_PREFERENCE_ORDER: MealPreferenceOption[] = [
  "vegetarian",
  "kosher",
  "gluten intolerant",
  "child",
];

export function mealPreferenceLabel(value: MealPreferenceOption): string {
  return MEAL_PREFERENCE_LABELS[value] || value.charAt(0).toUpperCase() + value.slice(1);
}

export interface WidgetAction {
  type: string;
  payload: SetMealPreferencePayload;
  handler: "client" | "server";
}

export interface IconWidget {
  type: "Icon";
  name: string;
  color?: string;
}

export interface TextWidget {
  type: "Text";
  value: string;
  weight?: string;
  color?: string;
}

export interface RowWidget {
  type: "Row";
  gap?: number;
  children: (IconWidget | TextWidget)[];
}

export interface ListViewItemWidget {
  type: "ListViewItem";
  key: string;
  onClickAction?: WidgetAction | null;
  children: RowWidget[];
}

export interface ListViewWidget {
  type: "ListView";
  key: string;
  children: ListViewItemWidget[];
}

function createSetMealPreferenceAction(meal: MealPreferenceOption): WidgetAction {
  return {
    type: SET_MEAL_PREFERENCE_ACTION_TYPE,
    payload: { meal },
    handler: "client",
  };
}

export function buildMealPreferenceWidget(
  selected: MealPreferenceOption | null = null
): ListViewWidget {
  const items: ListViewItemWidget[] = [];

  for (const value of MEAL_PREFERENCE_ORDER) {
    const label = mealPreferenceLabel(value);
    const emphasized = value === selected;
    const actionConfig = selected === null ? createSetMealPreferenceAction(value) : null;

    items.push({
      type: "ListViewItem",
      key: `meal-${value}`,
      onClickAction: actionConfig,
      children: [
        {
          type: "Row",
          gap: 2,
          children: [
            {
              type: "Icon",
              name: emphasized ? "check" : "empty-circle",
              color: "secondary",
            },
            {
              type: "Text",
              value: label,
              weight: emphasized ? "semibold" : "medium",
              color: emphasized ? "emphasis" : undefined,
            },
          ],
        },
      ],
    });
  }

  return {
    type: "ListView",
    key: "meal-preference-list",
    children: items,
  };
}

