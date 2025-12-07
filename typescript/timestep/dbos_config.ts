/** DBOS configuration for Timestep workflows. */

import { DBOS } from '@dbos-inc/dbos-sdk';
import { getPgliteDir } from './app_dir.ts';
import * as path from 'path';

let dbosConfigured = false;
let dbosLaunched = false;
let pgliteServer: any = null;
let pgliteDb: any = null;

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
    
    // Start PGLite socket server
    const { PGlite } = await import('@electric-sql/pglite');
    const { PGLiteSocketServer } = await import('@electric-sql/pglite-socket');
    
    // Create PGLite instance
    pgliteDb = new PGlite(dbPath, { dataDir: dbPath });
    await pgliteDb.waitReady;
    
    // Use Unix socket for better performance (or TCP if on Windows)
    const isWindows = typeof process !== 'undefined' && process.platform === 'win32';
    const socketPath = isWindows ? undefined : path.join(pglitePath, '.s.PGSQL.5432');
    const port = isWindows ? 5432 : undefined;
    
    pgliteServer = new PGLiteSocketServer({
      db: pgliteDb,
      ...(socketPath ? { path: socketPath } : { port, host: '127.0.0.1' }),
    });
    
    await pgliteServer.start();
    
    // Construct PostgreSQL connection string pointing to the socket server
    if (socketPath) {
      // Unix socket connection
      dbUrl = `postgresql://postgres:postgres@/postgres?host=${path.dirname(socketPath)}&sslmode=disable`;
    } else {
      // TCP connection
      dbUrl = `postgresql://postgres:postgres@127.0.0.1:${port}/postgres?sslmode=disable`;
    }
  }
  
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
    await configureDBOS();
  }
  
  if (!dbosLaunched) {
    await DBOS.launch();
    dbosLaunched = true;
  }
}

/**
 * Clean up PGLite socket server and database connections.
 * Call this when shutting down the application.
 */
export async function cleanupDBOS(): Promise<void> {
  if (pgliteServer) {
    await pgliteServer.stop();
    pgliteServer = null;
  }
  if (pgliteDb) {
    await pgliteDb.close();
    pgliteDb = null;
  }
  dbosConfigured = false;
  dbosLaunched = false;
}

