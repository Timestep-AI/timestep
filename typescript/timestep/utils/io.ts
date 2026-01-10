/** I/O utilities. */

import { writeFileSync, mkdirSync } from 'fs';
import { dirname } from 'path';

export function now(): number {
  return Date.now() / 1000;
}

export function clamp01(x: number): number {
  return Math.max(0.0, Math.min(1.0, x));
}

export function writeJson(path: string, obj: any): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(obj, null, 2), 'utf-8');
}
