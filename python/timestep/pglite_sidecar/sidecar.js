#!/usr/bin/env node
/**
 * PGlite Sidecar - Long-lived Node.js process for PGlite queries
 * 
 * Communicates via JSON-RPC over stdin/stdout:
 * - Request: {"jsonrpc": "2.0", "id": 1, "method": "query", "params": {"sql": "...", "params": [...]}}
 * - Response: {"jsonrpc": "2.0", "id": 1, "result": {...}} or {"jsonrpc": "2.0", "id": 1, "error": {...}}
 */

// Try to load PGlite from various locations
let PGlite;
let PGLiteSocketServer;
try {
  PGlite = require('@electric-sql/pglite').PGlite;
  try {
    PGLiteSocketServer = require('@electric-sql/pglite-socket').PGLiteSocketServer;
  } catch (e) {
    // Socket server is optional - only needed for DBOS
    PGLiteSocketServer = null;
  }
} catch (e) {
  // Try to find it in common locations
  const path = require('path');
  const fs = require('fs');
  
  const possiblePaths = [
    path.join(process.env.HOME || process.env.USERPROFILE || '', '.npm', 'global', 'node_modules', '@electric-sql', 'pglite'),
    path.join('/usr', 'lib', 'node_modules', '@electric-sql', 'pglite'),
    path.join('/usr', 'local', 'lib', 'node_modules', '@electric-sql', 'pglite'),
  ];
  
  let found = false;
  for (const pglitePath of possiblePaths) {
    if (fs.existsSync(path.join(pglitePath, 'package.json'))) {
      try {
        PGlite = require(pglitePath).PGlite;
        found = true;
        // Try to load socket server from same location
        try {
          const socketPath = path.join(path.dirname(pglitePath), 'pglite-socket');
          if (fs.existsSync(path.join(socketPath, 'package.json'))) {
            PGLiteSocketServer = require(socketPath).PGLiteSocketServer;
          }
        } catch (e2) {
          PGLiteSocketServer = null;
        }
        break;
      } catch (e2) {
        // Continue trying
      }
    }
  }
  
  if (!found) {
    console.error(JSON.stringify({
      jsonrpc: "2.0",
      error: {
        code: -32603,
        message: "@electric-sql/pglite is not installed. Install it with: npm install -g @electric-sql/pglite"
      }
    }));
    process.exit(1);
  }
}

// Get PGLite data directory from environment or command line
const pglitePath = process.env.PGLITE_PATH || process.argv[2];

if (!pglitePath) {
  console.error(JSON.stringify({
    jsonrpc: "2.0",
    error: {
      code: -32600,
      message: "PGLITE_PATH environment variable or path argument required"
    }
  }));
  process.exit(1);
}

let db = null;
let socketServer = null;
let isReady = false;
const initPromise = (async () => {
  try {
    db = new PGlite(pglitePath);
    await db.waitReady;
    isReady = true;
    
    // Send ready notification
    console.error(JSON.stringify({
      jsonrpc: "2.0",
      method: "ready",
      params: { status: "ready" }
    }));
  } catch (error) {
    console.error(JSON.stringify({
      jsonrpc: "2.0",
      error: {
        code: -32000,
        message: `Failed to initialize PGlite: ${error.message}`
      }
    }));
    process.exit(1);
  }
})();

// Buffer for incomplete JSON lines
let buffer = '';

// Read from stdin line by line (JSON-RPC messages)
process.stdin.setEncoding('utf8');

process.stdin.on('data', async (chunk) => {
  buffer += chunk;
  const lines = buffer.split('\n');
  buffer = lines.pop() || ''; // Keep incomplete line in buffer
  
  for (const line of lines) {
    if (!line.trim()) continue;
    
    try {
      const request = JSON.parse(line);
      await handleRequest(request);
    } catch (error) {
      // Send parse error response
      sendError(null, -32700, 'Parse error', error.message);
    }
  }
});

process.stdin.on('end', async () => {
  if (socketServer) {
    try {
      await socketServer.stop();
    } catch (error) {
      // Ignore close errors
    }
  }
  if (db) {
    try {
      await db.close();
    } catch (error) {
      // Ignore close errors
    }
  }
  process.exit(0);
});

// Handle shutdown signals
process.on('SIGINT', async () => {
  if (socketServer) {
    try {
      await socketServer.stop();
    } catch (error) {
      // Ignore close errors
    }
  }
  if (db) {
    try {
      await db.close();
    } catch (error) {
      // Ignore close errors
    }
  }
  process.exit(0);
});

process.on('SIGTERM', async () => {
  if (socketServer) {
    try {
      await socketServer.stop();
    } catch (error) {
      // Ignore close errors
    }
  }
  if (db) {
    try {
      await db.close();
    } catch (error) {
      // Ignore close errors
    }
  }
  process.exit(0);
});

async function handleRequest(request) {
  // Wait for initialization if not ready
  if (!isReady) {
    await initPromise;
  }
  
  const { jsonrpc, id, method, params } = request;
  
  if (jsonrpc !== '2.0') {
    sendError(id, -32600, 'Invalid Request', 'jsonrpc must be "2.0"');
    return;
  }
  
  if (!id) {
    sendError(null, -32600, 'Invalid Request', 'id is required');
    return;
  }
  
  try {
    let result;
    
    switch (method) {
      case 'query':
        if (!params || typeof params.sql !== 'string') {
          sendError(id, -32602, 'Invalid params', 'sql parameter is required');
          return;
        }
        result = await db.query(params.sql, params.params || []);
        sendResponse(id, {
          rows: result.rows || [],
          rowCount: result.rowCount || (result.rows ? result.rows.length : 0)
        });
        break;
        
      case 'execute':
        if (!params || typeof params.sql !== 'string') {
          sendError(id, -32602, 'Invalid params', 'sql parameter is required');
          return;
        }
        result = await db.query(params.sql, params.params || []);
        sendResponse(id, {
          rowCount: result.rowCount || 0
        });
        break;
        
      case 'close':
        if (db) {
          await db.close();
          db = null;
          isReady = false;
        }
        sendResponse(id, { closed: true });
        // Exit after close
        setTimeout(() => process.exit(0), 100);
        break;
        
      case 'ping':
        sendResponse(id, { pong: true, ready: isReady });
        break;
        
      case 'start_socket_server':
        if (!PGLiteSocketServer) {
          sendError(id, -32603, 'Not available', '@electric-sql/pglite-socket is not installed. Install it with: npm install -g @electric-sql/pglite-socket');
          return;
        }
        if (socketServer) {
          sendResponse(id, { 
            started: true, 
            port: socketServer.server?.address()?.port || params?.port || 5432,
            path: socketServer.server?.path || null
          });
          return;
        }
        try {
          const port = params?.port || 0; // 0 = auto-assign
          const host = params?.host || '127.0.0.1';
          const socketPath = params?.path || null;
          
          socketServer = new PGLiteSocketServer({
            db: db,
            ...(socketPath ? { path: socketPath } : { port: port, host: host }),
          });
          
          await socketServer.start();
          
          const actualPort = socketServer.server?.address()?.port || port;
          const actualPath = socketServer.server?.path || socketPath;
          
          sendResponse(id, { 
            started: true, 
            port: actualPort,
            path: actualPath,
            connectionString: socketPath 
              ? `postgresql://postgres:postgres@/postgres?host=${require('path').dirname(actualPath)}&sslmode=disable`
              : `postgresql://postgres:postgres@${host}:${actualPort}/postgres?sslmode=disable`
          });
        } catch (error) {
          sendError(id, -32000, 'Socket server error', error.message);
        }
        break;
        
      case 'stop_socket_server':
        if (socketServer) {
          try {
            await socketServer.stop();
            socketServer = null;
            sendResponse(id, { stopped: true });
          } catch (error) {
            sendError(id, -32000, 'Socket server error', error.message);
          }
        } else {
          sendResponse(id, { stopped: true, wasRunning: false });
        }
        break;
        
      default:
        sendError(id, -32601, 'Method not found', `Unknown method: ${method}`);
    }
  } catch (error) {
    sendError(id, -32000, 'Execution error', error.message);
  }
}

function sendResponse(id, result) {
  const response = {
    jsonrpc: '2.0',
    id,
    result
  };
  console.log(JSON.stringify(response));
}

function sendError(id, code, message, data) {
  const response = {
    jsonrpc: '2.0',
    id,
    error: {
      code,
      message,
      ...(data && { data })
    }
  };
  console.log(JSON.stringify(response));
}

