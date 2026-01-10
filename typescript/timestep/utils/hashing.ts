/** Hashing utilities. */

import { createHash } from 'crypto';

export function stableHash(obj: any): string {
  const s = JSON.stringify(obj, Object.keys(obj).sort());
  return createHash('sha256').update(s, 'utf-8').digest('hex').slice(0, 12);
}
