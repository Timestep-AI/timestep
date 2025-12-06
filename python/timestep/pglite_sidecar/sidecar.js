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
try {
  PGlite = require('@electric-sql/pglite').PGlite;
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

