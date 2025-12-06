"""Cross-platform application directory utilities.

Based on Click's get_app_dir implementation for consistency with Typer/Click conventions.
"""

import os
import sys
from pathlib import Path


def _posixify(name: str) -> str:
    """Convert app name to POSIX-friendly format."""
    return "-".join(name.split()).lower()


def get_app_dir(
    app_name: str = "timestep",
    roaming: bool = True,
    force_posix: bool = False
) -> Path:
    """
    Get the application directory for storing configuration and data files.
    
    Based on Click's get_app_dir implementation. Returns platform-appropriate directories:
    - Windows (roaming): %APPDATA%/timestep
    - Windows (not roaming): %LOCALAPPDATA%/timestep
    - macOS: ~/Library/Application Support/timestep
    - Unix/Linux: ~/.config/timestep (or $XDG_CONFIG_HOME/timestep)
    - POSIX mode: ~/.timestep
    
    Args:
        app_name: Application name (default: "timestep")
        roaming: On Windows, use roaming profile (default: True)
        force_posix: Use POSIX-style ~/.app-name instead of platform defaults
    
    Returns:
        Path to the application directory (created if it doesn't exist)
    """
    WIN = os.name == "nt"
    
    if WIN:
        key = "APPDATA" if roaming else "LOCALAPPDATA"
        folder = os.environ.get(key)
        if folder is None:
            folder = os.path.expanduser("~")
        app_path = Path(folder) / app_name
    elif force_posix:
        app_path = Path(os.path.expanduser(f"~/.{_posixify(app_name)}"))
    elif sys.platform == "darwin":  # macOS
        app_path = Path(os.path.expanduser("~/Library/Application Support")) / app_name
    else:  # Unix/Linux
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config_home:
            config_home = Path(xdg_config_home)
        else:
            config_home = Path.home() / ".config"
        app_path = config_home / _posixify(app_name)
    
    app_path.mkdir(parents=True, exist_ok=True)
    return app_path


def get_pglite_dir(session_id: str | None = None) -> Path:
    """
    Get the PGLite database directory for a session.
    
    Args:
        session_id: Optional session ID to create a unique database per session
    
    Returns:
        Path to the PGLite database directory
    """
    app_dir = get_app_dir()
    if session_id:
        return app_dir / "pglite" / f"db_{session_id}"
    return app_dir / "pglite" / "default"

