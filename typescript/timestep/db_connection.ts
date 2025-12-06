/** Database connection management for Timestep. */

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
    this.connectionString =
      options.connectionString || Deno.env.get('TIMESTEP_DB_URL');
    this.usePglite =
      options.usePglite ||
      Deno.env.get('TIMESTEP_USE_PGLITE')?.toLowerCase() === 'true';
    this.pglitePath =
      options.pglitePath || Deno.env.get('TIMESTEP_PGLITE_PATH') || './pglite_data';
  }

  async connect(): Promise<boolean> {
    // Try PostgreSQL first if connection string is provided
    if (this.connectionString && !this.usePglite) {
      try {
        return await this.connectPostgreSQL();
      } catch (e) {
        // Fall through to try PGLite
      }
    }

    // Try PGLite if enabled
    if (this.usePglite) {
      try {
        return await this.connectPglite();
      } catch (e) {
        // Connection failed
      }
    }

    return false;
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

      this.connection = new PGlite(this.pglitePath);
      this.dbType = DatabaseType.PGLITE;

      // Initialize database (creates if doesn't exist)
      await this.connection.waitReady;

      // Test connection
      await this.connection.query('SELECT 1');
      return true;
    } catch (e: any) {
      if (e.code === 'MODULE_NOT_FOUND') {
        throw new Error(
          '@electric-sql/pglite is required for PGLite support. Install it with: npm install @electric-sql/pglite'
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

