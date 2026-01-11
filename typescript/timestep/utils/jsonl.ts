/** JSONL file I/O utilities. */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { dirname } from 'path';
import type { JSON } from '../core/types';

export function* readJsonl(path: string): Generator<JSON> {
  const content = readFileSync(path, 'utf-8');
  const lines = content.split('\n');
  
  for (let lineNo = 0; lineNo < lines.length; lineNo++) {
    const line = lines[lineNo].trim();
    if (!line) continue;
    
    try {
      yield JSON.parse(line);
    } catch (e) {
      throw new Error(`Invalid JSON on line ${lineNo + 1} in ${path}: ${e}`);
    }
  }
}

export function writeJsonl(path: string, rows: JSON[]): void {
  mkdirSync(dirname(path), { recursive: true });
  const content = rows.map(r => JSON.stringify(r)).join('\n') + '\n';
  writeFileSync(path, content, 'utf-8');
}
