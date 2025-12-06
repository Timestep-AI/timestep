/** Cross-platform application directory utilities.
 *
 * Based on Click's get_app_dir implementation for consistency with Typer/Click conventions.
 */

import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

function _posixify(name: string): string {
  /** Convert app name to POSIX-friendly format. */
  return name.split(/\s+/).join('-').toLowerCase();
}

export interface GetAppDirOptions {
  /** Application name (default: "timestep") */
  appName?: string;
  /** On Windows, use roaming profile (default: true) */
  roaming?: boolean;
  /** Use POSIX-style ~/.app-name instead of platform defaults */
  forcePosix?: boolean;
}

/**
 * Get the application directory for storing configuration and data files.
 *
 * Based on Click's get_app_dir implementation. Returns platform-appropriate directories:
 * - Windows (roaming): %APPDATA%/timestep
 * - Windows (not roaming): %LOCALAPPDATA%/timestep
 * - macOS: ~/Library/Application Support/timestep
 * - Unix/Linux: ~/.config/timestep (or $XDG_CONFIG_HOME/timestep)
 * - POSIX mode: ~/.timestep
 *
 * @param options Configuration options
 * @returns Path to the application directory (created if it doesn't exist)
 */
export function getAppDir(options: GetAppDirOptions = {}): string {
  const {
    appName = 'timestep',
    roaming = true,
    forcePosix = false,
  } = options;

  const isWindows = process.platform === 'win32';

  let appPath: string;

  if (isWindows) {
    const key = roaming ? 'APPDATA' : 'LOCALAPPDATA';
    const folder = process.env[key] || os.homedir();
    appPath = path.join(folder, appName);
  } else if (forcePosix) {
    appPath = path.join(os.homedir(), `.${_posixify(appName)}`);
  } else if (process.platform === 'darwin') {
    // macOS
    appPath = path.join(os.homedir(), 'Library', 'Application Support', appName);
  } else {
    // Unix/Linux
    const xdgConfigHome = process.env.XDG_CONFIG_HOME;
    const configHome = xdgConfigHome || path.join(os.homedir(), '.config');
    appPath = path.join(configHome, _posixify(appName));
  }

  // Ensure directory exists
  if (!fs.existsSync(appPath)) {
    fs.mkdirSync(appPath, { recursive: true });
  }

  return appPath;
}

export interface GetPgliteDirOptions {
  /** Optional session ID to create a unique database per session */
  sessionId?: string;
  /** Application name (default: "timestep") */
  appName?: string;
}

/**
 * Get the PGLite database directory for a session.
 *
 * @param options Configuration options
 * @returns Path to the PGLite database directory
 */
export function getPgliteDir(options: GetPgliteDirOptions = {}): string {
  const { sessionId, appName } = options;
  const appDir = getAppDir({ appName });
  const pgliteDir = path.join(appDir, 'pglite');

  if (sessionId) {
    return path.join(pgliteDir, `db_${sessionId}`);
  }
  return path.join(pgliteDir, 'default');
}

