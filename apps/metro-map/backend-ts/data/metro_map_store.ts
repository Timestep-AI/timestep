/**
 * MetroMapStore - Data models and store for the metro map demo.
 */

export interface Station {
  id: string;
  name: string;
  x: number;
  y: number;
  description: string;
  lines: string[];
}

export interface Line {
  id: string;
  name: string;
  color: string;
  orientation: "horizontal" | "vertical";
  stations: string[];
}

export interface MetroMap {
  id: string;
  name: string;
  summary: string;
  stations: Station[];
  lines: Line[];
}

export class MetroMapStore {
  private map!: MetroMap;
  private stationLookup: Map<string, Station> = new Map();
  private lineLookup: Map<string, Line> = new Map();

  constructor(dataDir: string) {
    const mapPath = `${dataDir}/metro_map.json`;
    const mapData = JSON.parse(Deno.readTextFileSync(mapPath));
    this.setMap(mapData as MetroMap);
  }

  // -- Queries --
  getMap(): MetroMap {
    return this.map;
  }

  listLines(): Line[] {
    return [...this.map.lines];
  }

  listStations(): Station[] {
    return [...this.map.stations];
  }

  findStation(stationId: string): Station | null {
    return this.stationLookup.get(stationId) || null;
  }

  findLine(lineId: string): Line | null {
    return this.lineLookup.get(lineId) || null;
  }

  stationsForLine(lineId: string): Station[] {
    const line = this.lineLookup.get(lineId);
    if (!line) return [];
    return line.stations
      .map((id) => this.stationLookup.get(id))
      .filter((s): s is Station => s !== undefined);
  }

  dumpForClient(): Record<string, unknown> {
    return { ...this.map };
  }

  // -- Mutations --
  setMap(map: MetroMap): void {
    this.map = map;
    this.stationLookup.clear();
    this.lineLookup.clear();
    for (const station of map.stations) {
      this.stationLookup.set(station.id, station);
    }
    for (const line of map.lines) {
      this.lineLookup.set(line.id, line);
    }
  }

  addStation(
    stationName: string,
    lineId: string,
    append: boolean = true,
    description: string = ""
  ): [MetroMap, Station] {
    const normalizedLineId = this.normalizeId(lineId);
    const line = this.lineLookup.get(lineId) || this.lineLookup.get(normalizedLineId);
    if (!line || line.stations.length === 0) {
      throw new Error(`Line '${lineId}' is not found.`);
    }

    const stationId = this.nextStationId(stationName);
    const insertionIndex = append ? line.stations.length : 0;
    const [x, y] = this.getCoordinatesForNewStation(line, append);

    const station: Station = {
      id: stationId,
      name: stationName,
      x,
      y,
      description,
      lines: [line.id],
    };

    this.map.stations.push(station);
    this.stationLookup.set(station.id, station);
    line.stations.splice(insertionIndex, 0, station.id);

    return [this.map, station];
  }

  // -- Helpers --
  private normalizeId(value: string, fallback: string = "id"): string {
    let slug = value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    if (!slug) slug = fallback;
    return slug;
  }

  private nextStationId(stationName: string): string {
    const base = this.normalizeId(stationName);
    let candidate = base;
    let counter = 2;
    while (this.stationLookup.has(candidate)) {
      candidate = `${base}-${counter}`;
      counter++;
    }
    return candidate;
  }

  private getCoordinatesForNewStation(line: Line, append: boolean): [number, number] {
    if (append) {
      const prevId = line.stations[line.stations.length - 1];
      const prev = this.stationLookup.get(prevId);
      const [prevX, prevY] = prev ? [prev.x, prev.y] : [0, 0];
      return line.orientation === "horizontal" ? [prevX + 1, prevY] : [prevX, prevY + 1];
    }

    const nextId = line.stations[0];
    const next = this.stationLookup.get(nextId);
    const [nextX, nextY] = next ? [next.x, next.y] : [0, 0];
    return line.orientation === "horizontal" ? [nextX - 1, nextY] : [nextX, nextY - 1];
  }
}

