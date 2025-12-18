// Debug stub for Deno - avoids CommonJS compatibility issues
const noop = () => {};
const debug = (namespace: string) => {
  const debugFn = (...args: unknown[]) => {
    if (Deno.env.get("DEBUG")?.includes(namespace) || Deno.env.get("DEBUG") === "*") {
      console.log(`[${namespace}]`, ...args);
    }
  };
  debugFn.enabled = false;
  debugFn.namespace = namespace;
  debugFn.extend = (sub: string) => debug(`${namespace}:${sub}`);
  debugFn.log = noop;
  return debugFn;
};

export default debug;
export { debug };

