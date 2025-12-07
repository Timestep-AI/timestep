/** DBOS configuration for Timestep workflows. */

import { DBOS } from '@dbos-inc/dbos-sdk';
import { getPgliteDir } from './app_dir.ts';
import * as path from 'path';

class DBOSContext {
  private config: { name: string; systemDatabaseUrl: string } | null = null;
  private configured = false;
  private launched = false;
  private pgliteServer: any = null;
  private pgliteDb: any = null;

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

  getPgliteServer(): any {
    return this.pgliteServer;
  }

  setPgliteServer(server: any): void {
    this.pgliteServer = server;
  }

  getPgliteDb(): any {
    return this.pgliteDb;
  }

  setPgliteDb(db: any): void {
    this.pgliteDb = db;
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

export interface ConfigureDBOSOptions {
  /** Application name for DBOS (default: "timestep") */
  name?: string;
  /** Optional system database URL. If not provided, uses PG_CONNECTION_URI environment variable, or PGLite */
  systemDatabaseUrl?: string;
}

/**
 * Configure DBOS for Timestep workflows.
 * 
 * Uses PG_CONNECTION_URI for the system database if available,
 * otherwise uses PGLite via socket server (same as RunStateStore).
 * 
 * @param options Configuration options
 */
export async function configureDBOS(options: ConfigureDBOSOptions = {}): Promise<void> {
  const { name = 'timestep', systemDatabaseUrl } = options;
  
  // Get system database URL from parameter, env var, or default to PGLite
  const env = typeof process !== 'undefined' ? process.env : {};
  let dbUrl = systemDatabaseUrl || env['PG_CONNECTION_URI'];
  
  // If no connection string provided, start PGLite socket server
  if (!dbUrl) {
    // Use a dedicated PGLite database for DBOS system database
    const pglitePath = getPgliteDir({ appName: name });
    const dbPath = path.join(pglitePath, 'dbos_system');
    
    // Ensure parent directory exists
    const fs = await import('fs/promises');
    try {
      await fs.mkdir(path.dirname(dbPath), { recursive: true });
    } catch (e: any) {
      // Ignore if directory already exists
      if (e.code !== 'EEXIST') {
        throw e;
      }
    }
    
    // Start PGLite socket server
    const { PGlite } = await import('@electric-sql/pglite');
    const { PGLiteSocketServer } = await import('@electric-sql/pglite-socket');
    const { uuid_ossp } = await import('@electric-sql/pglite/contrib/uuid_ossp');
    
    // Create PGLite instance with uuid-ossp extension (required by DBOS)
    const pgliteDb = new PGlite(dbPath, { 
      dataDir: dbPath,
      extensions: { uuid_ossp }
    });
    await pgliteDb.waitReady;
    dbosContext.setPgliteDb(pgliteDb);
    
    // Use TCP connection for simplicity and compatibility
    // Port 0 means let the OS assign an available port
    const port = 0;
    
    const pgliteServer = new PGLiteSocketServer({
      db: pgliteDb,
      port: port,
      host: '127.0.0.1',
    });
    
    await pgliteServer.start();
    dbosContext.setPgliteServer(pgliteServer);
    
    // Wait for server to initialize (PGLite socket server needs a moment)
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Get the actual port that was assigned
    const server = (dbosContext.getPgliteServer() as any).server;
    if (!server) {
      throw new Error('Server object not found after start()');
    }
    
    const actualPort = server.address()?.port;
    if (!actualPort) {
      throw new Error('Failed to get port from PGLite socket server');
    }
    
    // Test the connection before using it
    // Note: PGLite socket server only supports one connection at a time,
    // so we test and immediately close to ensure DBOS can connect
    try {
      const pg = await import('pg');
      const { Client } = pg;
      const testClient = new Client({
        host: '127.0.0.1',
        port: actualPort,
        user: 'postgres',
        password: 'postgres',
        database: 'postgres',
        ssl: false,
        connectionTimeoutMillis: 10000,
      });
      await testClient.connect();
      await testClient.query('SELECT 1');
      await testClient.end();
      // Wait for connection to fully close before DBOS connects
      // PGLite socket server only supports one connection at a time
      await new Promise(resolve => setTimeout(resolve, 500));
    } catch (e: any) {
      throw new Error(`PGLite socket server connection test failed: ${e.message}`);
    }
    
    // Construct PostgreSQL connection string pointing to the socket server
    dbUrl = `postgresql://postgres:postgres@127.0.0.1:${actualPort}/postgres?sslmode=disable`;
    console.log(`PGLite socket server listening on port ${actualPort}`);
    console.log(`DBOS will use connection string: postgresql://postgres:postgres@127.0.0.1:${actualPort}/postgres?sslmode=disable`);
  }
  
  // DBOS will use the same database but different schema (dbos schema)
  const config = {
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
    
    // Ensure config is set in DBOS (in case it was cleared or reset)
    const config = dbosContext.getConnectionString();
    if (config) {
      DBOS.setConfig({
        name: 'timestep',
        systemDatabaseUrl: config,
      });
      console.log(`Re-set DBOS config with connection string: ${config}`);
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
 * Clean up PGLite socket server and database connections.
 * Call this when shutting down the application.
 */
export async function cleanupDBOS(): Promise<void> {
  const pgliteServer = dbosContext.getPgliteServer();
  if (pgliteServer) {
    await pgliteServer.stop();
    dbosContext.setPgliteServer(null);
  }
  const pgliteDb = dbosContext.getPgliteDb();
  if (pgliteDb) {
    await pgliteDb.close();
    dbosContext.setPgliteDb(null);
  }
  dbosContext.setConfigured(false);
  dbosContext.setLaunched(false);
}

