/** Database connection management for Timestep. */

export enum DatabaseType {
  POSTGRESQL = 'postgresql',
  NONE = 'none',
}

export interface DatabaseConnectionOptions {
  connectionString: string;
}

export class DatabaseConnection {
  private connection: any = null;
  private dbType: DatabaseType = DatabaseType.NONE;
  private connectionString: string;

  constructor(options: DatabaseConnectionOptions) {
    if (!options.connectionString) {
      throw new Error('connectionString is required for DatabaseConnection');
    }
    this.connectionString = options.connectionString;
  }

  async connect(): Promise<boolean> {
    /**
     * Connect to database.
     */
    try {
      return await this.connectPostgreSQL();
    } catch (e) {
      throw new Error(`Failed to connect to PostgreSQL: ${e}`);
    }
  }

  private async connectPostgreSQL(): Promise<boolean> {
    try {
      // Dynamic import to avoid requiring pg at module load time
      // pg is a CommonJS module, so in ESM it's wrapped in a default export
      const pgModule = await import('pg');
      // Handle CommonJS module loaded via ESM: default export contains the actual module
      const pg = (pgModule as any).default || pgModule;
      const Pool = pg.Pool;
      
      if (!Pool || typeof Pool !== 'function') {
        throw new Error(
          `Pool is not a constructor. ` +
          `pg keys: ${JSON.stringify(Object.keys(pg))}, ` +
          `pgModule keys: ${JSON.stringify(Object.keys(pgModule))}, ` +
          `Pool type: ${typeof Pool}`
        );
      }

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

  async disconnect(): Promise<void> {
    if (this.connection) {
      if (this.dbType === DatabaseType.POSTGRESQL) {
        await this.connection.end();
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
    } else {
      throw new Error(`Query not implemented for ${this.dbType}`);
    }
  }
}
