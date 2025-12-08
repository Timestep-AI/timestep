/** DBOS configuration for Timestep workflows. */

import { DBOS } from '@dbos-inc/dbos-sdk';

class DBOSContext {
  private config: { name: string; systemDatabaseUrl: string } | null = null;
  private configured = false;
  private launched = false;

  getConnectionString(): string | undefined {
    if (this.config) {
      return this.config.systemDatabaseUrl;
    }
    return undefined;
  }

  get isConfigured(): boolean {
    return this.configured;
  }

  get isLaunched(): boolean {
    return this.launched;
  }

  setConfig(config: { name: string; systemDatabaseUrl: string }): void {
    this.config = config;
  }

  setConfigured(value: boolean): void {
    this.configured = value;
  }

  setLaunched(value: boolean): void {
    this.launched = value;
  }
}

// Singleton instance
const dbosContext = new DBOSContext();

/**
 * Get the DBOS connection string if configured.
 * 
 * @returns Connection string or undefined if not configured
 */
export function getDBOSConnectionString(): string | undefined {
  return dbosContext.getConnectionString();
}

/**
 * Check if DBOS has been launched.
 * 
 * @returns True if DBOS has been launched, false otherwise
 */
export function isDBOSLaunched(): boolean {
  return dbosContext.isLaunched;
}

export interface ConfigureDBOSOptions {
  /** Application name for DBOS (default: "timestep") */
  name?: string;
  /** Optional system database URL. If not provided, uses PG_CONNECTION_URI environment variable */
  systemDatabaseUrl?: string;
}

/**
 * Configure DBOS for Timestep workflows.
 * 
 * Uses PG_CONNECTION_URI environment variable for the system database.
 * For tests, run 'make test-setup' to start the test database.
 * 
 * @param options Configuration options
 */
export async function configureDBOS(options: ConfigureDBOSOptions = {}): Promise<void> {
  const { name = 'timestep', systemDatabaseUrl } = options;
  
  // Get system database URL from parameter or env var
  const env = typeof process !== 'undefined' ? process.env : {};
  const dbUrl = systemDatabaseUrl || env['PG_CONNECTION_URI'];
  
  if (!dbUrl) {
    throw new Error(
      "PG_CONNECTION_URI not set. Run 'make test-setup' to start the test database, " +
      "or set PG_CONNECTION_URI environment variable to your PostgreSQL connection string."
    );
  }
  
  // DBOS will use the same database but different schema (dbos schema)
  const config: any = {
    name,
    systemDatabaseUrl: dbUrl,
  };
  
  console.log(`Setting DBOS config with systemDatabaseUrl: ${config.systemDatabaseUrl}`);
  DBOS.setConfig(config);
  dbosContext.setConfig(config);
  dbosContext.setConfigured(true);
}

/**
 * Ensure DBOS is launched. Safe to call multiple times.
 * 
 * This should be called before using any DBOS workflows.
 */
export async function ensureDBOSLaunched(): Promise<void> {
  if (!dbosContext.isConfigured) {
    await configureDBOS();
  }
  
  // Always try to launch DBOS, regardless of our flag.
  // This ensures DBOS is actually launched even if our flag is incorrect.
  // DBOS.launch() should handle being called multiple times gracefully.
  try {
    await DBOS.launch();
    dbosContext.setLaunched(true);
  } catch (error: any) {
    // If launch fails, check if it's because DBOS is already launched
    const errorMsg = String(error).toLowerCase();
    if (errorMsg.includes('already') || errorMsg.includes('launched')) {
      // DBOS is already launched, just update our flag
      dbosContext.setLaunched(true);
    } else {
      // Some other error occurred - if our flag said it was launched,
      // reset it since launch failed
      if (dbosContext.isLaunched) {
        dbosContext.setLaunched(false);
      }
      throw error;
    }
  }
}

/**
 * Clean up DBOS resources.
 * Call this when shutting down the application.
 */
export async function cleanupDBOS(): Promise<void> {
  dbosContext.setConfigured(false);
  dbosContext.setLaunched(false);
}
