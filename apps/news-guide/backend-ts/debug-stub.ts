// debug-stub.ts - Deno-compatible stub for the 'debug' module
function debug(_namespace: string): (...args: unknown[]) => void {
  return (..._args: unknown[]) => {};
}
export default debug;

