// debug-stub.ts
// A simple stub for the 'debug' module to prevent 'require is not defined' errors in Deno.

function debug(_namespace: string): (...args: unknown[]) => void {
  return (..._args: unknown[]) => {};
}

export default debug;
export { debug };

