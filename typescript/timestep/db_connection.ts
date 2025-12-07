/** Database connection management for Timestep. */

import { getPgliteDir } from './app_dir.ts';

export enum DatabaseType {
  POSTGRESQL = 'postgresql',
  PGLITE = 'pglite',
  NONE = 'none',
}

export interface DatabaseConnectionOptions {
  connectionString?: string;
  usePglite?: boolean;
  pglitePath?: string;
}

export class DatabaseConnection {
  private connection: any = null;
  private dbType: DatabaseType = DatabaseType.NONE;
  private connectionString?: string;
  private usePglite: boolean;
  private pglitePath: string;

  constructor(options: DatabaseConnectionOptions = {}) {
    if (options.connectionString) {
      // PostgreSQL mode
      this.connectionString = options.connectionString;
      this.usePglite = false;
      this.pglitePath = '';
    } else if (options.usePglite) {
      // PGLite mode - path is required
      if (!options.pglitePath) {
        throw new Error('pglitePath is required when usePglite=true');
      }
      this.connectionString = undefined;
      this.usePglite = true;
      this.pglitePath = options.pglitePath;
    } else {
      // No configuration provided
      throw new Error(
        'Either connectionString must be provided for PostgreSQL, ' +
        'or usePglite=true with pglitePath for PGLite'
      );
    }
  }

  private _getDefaultPglitePath(): string {
    // Use app_dir module to match Python's get_app_dir() implementation
    return getPgliteDir();
  }

  async connect(): Promise<boolean> {
    /**
     * Connect to database.
     */
    if (this.connectionString && !this.usePglite) {
      try {
        return await this.connectPostgreSQL();
      } catch (e) {
        throw new Error(`Failed to connect to PostgreSQL: ${e}`);
      }
    }

    if (this.usePglite) {
      try {
        return await this.connectPglite();
      } catch (e) {
        throw new Error(`Failed to connect to PGLite: ${e}`);
      }
    }

    throw new Error('No connection configuration provided');
  }

  private async connectPostgreSQL(): Promise<boolean> {
    try {
      // Dynamic import to avoid requiring pg at module load time
      const pg = await import('pg');
      const { Pool } = pg;

      // Parse connection string and create pool
      this.connection = new Pool({
        connectionString: this.connectionString,
      });

      this.dbType = DatabaseType.POSTGRESQL;

      // Test connection
      await this.connection.query('SELECT 1');
      
      // Initialize schema
      const { initializeSchema } = await import('./schema.ts');
      await initializeSchema(this.connection);
      
      return true;
    } catch (e: any) {
      if (e.code === 'MODULE_NOT_FOUND' || e.message?.includes('Cannot find module')) {
        throw new Error(
          'pg is required for PostgreSQL support. Install it with: npm install pg @types/pg'
        );
      }
      throw new Error(`Failed to connect to PostgreSQL: ${e.message}`);
    }
  }

  private async connectPglite(): Promise<boolean> {
    try {
      // Dynamic import to avoid requiring @electric-sql/pglite at module load time
      const { PGlite } = await import('@electric-sql/pglite');
      const fs = await import('fs/promises');
      const path = await import('path');

      // Ensure directory exists
      const pgliteDir = path.dirname(this.pglitePath);
      try {
        await fs.mkdir(pgliteDir, { recursive: true });
      } catch (e: any) {
        // Ignore if directory already exists
        if (e.code !== 'EEXIST') {
          throw e;
        }
      }

      // Use 'idb' mode for better concurrency support
      // This allows multiple connections to the same database
      this.connection = new PGlite(this.pglitePath, { dataDir: this.pglitePath });
      this.dbType = DatabaseType.PGLITE;

      // Initialize database (creates if doesn't exist)
      await this.connection.waitReady;

      // Initialize schema (with error handling for concurrent access)
      try {
        const { initializeSchema } = await import('./schema.ts');
        await initializeSchema(this.connection);
      } catch (schemaError: any) {
        // If schema initialization fails due to concurrent access, that's okay
        // The table might already exist from another connection
        if (!schemaError.message?.includes('already exists') && 
            !schemaError.message?.includes('duplicate') &&
            !schemaError.message?.includes('Aborted')) {
          throw schemaError;
        }
      }

      // Test connection
      await this.connection.query('SELECT 1');
      return true;
    } catch (e: any) {
      if (e.code === 'MODULE_NOT_FOUND' || e.message?.includes('Cannot find module')) {
        throw new Error(
          '@electric-sql/pglite is required for PGLite support. Install it with: npm install @electric-sql/pglite'
        );
      }
      // Provide more helpful error message for WASM abort
      if (e.message?.includes('Aborted')) {
        throw new Error(
          `Failed to connect to PGLite: Database may be locked or corrupted. ` +
          `Try using a unique database path per test or ensure proper cleanup. ` +
          `Original error: ${e.message}`
        );
      }
      throw new Error(`Failed to connect to PGLite: ${e.message}`);
    }
  }

  async disconnect(): Promise<void> {
    if (this.connection) {
      if (this.dbType === DatabaseType.POSTGRESQL) {
        await this.connection.end();
      } else if (this.dbType === DatabaseType.PGLITE) {
        await this.connection.close();
      }
      this.connection = null;
      this.dbType = DatabaseType.NONE;
    }
  }

  get isConnected(): boolean {
    return this.connection !== null && this.dbType !== DatabaseType.NONE;
  }

  get connectionHandle(): any {
    if (!this.isConnected) {
      throw new Error('Database not connected. Call connect() first.');
    }
    return this.connection;
  }

  get type(): DatabaseType {
    return this.dbType;
  }

  async query(text: string, params?: any[]): Promise<any> {
    if (this.dbType === DatabaseType.POSTGRESQL) {
      return await this.connection.query(text, params);
    } else if (this.dbType === DatabaseType.PGLITE) {
      return await this.connection.query(text, params);
    } else {
      throw new Error(`Query not implemented for ${this.dbType}`);
    }
  }
}

