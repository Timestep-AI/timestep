/**
 * CatState class representing the state of a virtual cat.
 */

const STATUS_MIN = 0;
const STATUS_MAX = 10;
const COLOR_PATTERNS = ["black", "calico", "colorpoint", "tabby", "white"] as const;

function clamp(value: number): number {
  return Math.max(STATUS_MIN, Math.min(STATUS_MAX, value));
}

function randomChoice<T>(arr: readonly T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomBoolean(): boolean {
  return Math.random() < 0.5;
}

export class CatState {
  name: string;
  energy: number;
  happiness: number;
  cleanliness: number;
  age: number;
  colorPattern: string | null;
  updatedAt: Date;

  constructor(
    name: string = "Unnamed Cat",
    energy: number = 6,
    happiness: number = 6,
    cleanliness: number = 6,
    age: number = 2,
    colorPattern: string | null = null,
    updatedAt: Date = new Date()
  ) {
    this.name = name;
    this.energy = energy;
    this.happiness = happiness;
    this.cleanliness = cleanliness;
    this.age = age;
    this.colorPattern = colorPattern;
    this.updatedAt = updatedAt;
  }

  touch(): void {
    this.updatedAt = new Date();
  }

  feed(amount: number = 3): void {
    this.energy = clamp(this.energy + amount);
    this.happiness = clamp(this.happiness + 1);
    // Randomly deduct cleanliness on feed. Randomness makes it possible
    // for the cat status to reach 10 / 10 / 10.
    if (randomBoolean()) {
      this.cleanliness = clamp(this.cleanliness - 1);
    }
    this.touch();
  }

  play(boost: number = 2): void {
    this.happiness = clamp(this.happiness + boost);
    this.energy = clamp(this.energy - 1);
    // Randomly deduct cleanliness on play. Randomness makes it possible
    // for the cat status to reach 10 / 10 / 10.
    if (randomBoolean()) {
      this.cleanliness = clamp(this.cleanliness - 1);
    }
    this.touch();
  }

  clean(boost: number = 3): void {
    this.cleanliness = clamp(this.cleanliness + boost);
    // Randomly deduct happiness on clean. Randomness makes it possible
    // for the cat status to reach 10 / 10 / 10.
    if (randomBoolean()) {
      this.happiness = clamp(this.happiness - 1);
    }
    this.touch();
  }

  rename(value: string): void {
    console.log(`Renaming cat to ${value}`);
    this.name = value;
    if (!this.colorPattern) {
      console.log(`Choosing random color pattern for ${value}`);
      this.colorPattern = randomChoice(COLOR_PATTERNS);
      console.log(`Color pattern: ${this.colorPattern}`);
    }
    this.touch();
  }

  setAge(value: number | null): void {
    if (value && typeof value === "number") {
      this.age = Math.min(Math.max(value, 1), 15);
      this.touch();
    }
  }

  clone(): CatState {
    return new CatState(
      this.name,
      this.energy,
      this.happiness,
      this.cleanliness,
      this.age,
      this.colorPattern,
      new Date(this.updatedAt)
    );
  }

  toPayload(threadId?: string | null): Record<string, unknown> {
    const payload: Record<string, unknown> = {
      name: this.name,
      energy: this.energy,
      happiness: this.happiness,
      cleanliness: this.cleanliness,
      age: this.age,
      colorPattern: this.colorPattern,
      updatedAt: this.updatedAt.toISOString(),
    };
    if (threadId) {
      payload.threadId = threadId;
    }
    return payload;
  }
}

