/** DBOS configuration for Timestep workflows. */

import { DBOS } from '@dbos-inc/dbos-sdk';

let dbosConfigured = false;
let dbosLaunched = false;

export interface ConfigureDBOSOptions {
  /** Application name for DBOS (default: "timestep") */
  name?: string;
  /** Optional system database URL. If not provided, uses PG_CONNECTION_URI environment variable */
  systemDatabaseUrl?: string;
}

/**
 * Configure DBOS for Timestep workflows.
 * 
 * Uses PG_CONNECTION_URI for the system database if available,
 * otherwise uses a default SQLite database.
 * 
 * @param options Configuration options
 */
export function configureDBOS(options: ConfigureDBOSOptions = {}): void {
  const { name = 'timestep', systemDatabaseUrl } = options;
  
  // Get system database URL from parameter, env var, or default
  const env = typeof process !== 'undefined' ? process.env : {};
  const dbUrl = systemDatabaseUrl || env['PG_CONNECTION_URI'];
  
  // If we have a PostgreSQL connection string, use it for DBOS system database
  // DBOS will use the same database but different schema (dbos schema)
  DBOS.setConfig({
    name,
    systemDatabaseUrl: dbUrl,
  });
  
  dbosConfigured = true;
}

/**
 * Ensure DBOS is launched. Safe to call multiple times.
 * 
 * This should be called before using any DBOS workflows.
 */
export async function ensureDBOSLaunched(): Promise<void> {
  if (!dbosConfigured) {
    configureDBOS();
  }
  
  if (!dbosLaunched) {
    await DBOS.launch();
    dbosLaunched = true;
  }
}

