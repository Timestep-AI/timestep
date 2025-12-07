/** DBOS configuration for Timestep workflows. */

import { DBOS } from '@dbos-inc/dbos-sdk';
import { GenericContainer, StartedTestContainer, Wait } from 'testcontainers';

class DBOSContext {
  private config: { name: string; systemDatabaseUrl: string } | null = null;
  private configured = false;
  private launched = false;
  private postgresContainer: StartedTestContainer | null = null;

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

  getPostgresContainer(): StartedTestContainer | null {
    return this.postgresContainer;
  }

  setPostgresContainer(container: StartedTestContainer | null): void {
    this.postgresContainer = container;
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
  /** Optional system database URL. If not provided, uses PG_CONNECTION_URI environment variable, or Testcontainers PostgreSQL */
  systemDatabaseUrl?: string;
}

/**
 * Configure DBOS for Timestep workflows.
 * 
 * Uses PG_CONNECTION_URI for the system database if available,
 * otherwise starts a Testcontainers PostgreSQL instance.
 * 
 * @param options Configuration options
 */
export async function configureDBOS(options: ConfigureDBOSOptions = {}): Promise<void> {
  const { name = 'timestep', systemDatabaseUrl } = options;
  
  // Get system database URL from parameter, env var, or default to Testcontainers
  const env = typeof process !== 'undefined' ? process.env : {};
  let dbUrl = systemDatabaseUrl || env['PG_CONNECTION_URI'];
  
  // If no connection string provided, start Testcontainers PostgreSQL
  if (!dbUrl) {
    console.log('Starting Testcontainers PostgreSQL for DBOS...');
    const postgresContainer = await new GenericContainer('postgres:15')
      .withEnvironment({
        POSTGRES_DB: 'postgres',
        POSTGRES_USER: 'postgres',
        POSTGRES_PASSWORD: 'postgres',
      })
      .withExposedPorts(5432)
      .withWaitStrategy(Wait.forListeningPorts())
      .start();
    
    // Give PostgreSQL a moment to fully initialize after the port is listening
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    dbosContext.setPostgresContainer(postgresContainer);
    
    // Construct connection string
    const host = postgresContainer.getHost();
    const port = postgresContainer.getMappedPort(5432);
    dbUrl = `postgresql://postgres:postgres@${host}:${port}/postgres?sslmode=disable`;
    console.log(`Testcontainers PostgreSQL started: ${dbUrl}`);
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
  
  if (!dbosContext.isLaunched) {
    // Verify the config before launching
    const connectionString = dbosContext.getConnectionString();
    console.log(`DBOS config before launch - connection string: ${connectionString}`);
    if (!connectionString) {
      throw new Error('DBOS connection string is not set. Call configureDBOS() first.');
    }
    
    console.log('Calling DBOS.launch()...');
    try {
      await DBOS.launch();
      console.log('DBOS.launch() completed');
      dbosContext.setLaunched(true);
    } catch (error: any) {
      console.error('DBOS.launch() failed:', error);
      // Log the actual config DBOS is trying to use
      const currentConfig = dbosContext.getConnectionString();
      console.error(`DBOS was trying to connect to: ${currentConfig}`);
      throw error;
    }
  }
}

/**
 * Clean up Testcontainers PostgreSQL container.
 * Call this when shutting down the application.
 */
export async function cleanupDBOS(): Promise<void> {
  const postgresContainer = dbosContext.getPostgresContainer();
  if (postgresContainer) {
    await postgresContainer.stop();
    dbosContext.setPostgresContainer(null);
  }
  dbosContext.setConfigured(false);
  dbosContext.setLaunched(false);
}
