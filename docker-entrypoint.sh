#!/bin/sh
# Docker entrypoint script to create user if needed and run command

# Get UID and GID from environment (set by docker-compose)
USER_ID=${UID:-1000}
GROUP_ID=${GID:-1000}

# Check if user exists, if not create it
if ! id -u appuser >/dev/null 2>&1; then
    # Create group if it doesn't exist
    if ! getent group appgroup >/dev/null 2>&1; then
        addgroup -g "$GROUP_ID" appgroup 2>/dev/null || groupadd -g "$GROUP_ID" appgroup 2>/dev/null || true
    fi
    
    # Create user if it doesn't exist
    adduser -D -u "$USER_ID" -G appgroup appuser 2>/dev/null || \
    useradd -u "$USER_ID" -g appgroup -m appuser 2>/dev/null || true
fi

# Change ownership of working directory
chown -R appuser:appgroup /app 2>/dev/null || true

# Switch to the user and run the command
exec su-exec appuser "$@" 2>/dev/null || exec gosu appuser "$@" 2>/dev/null || exec runuser -u appuser -- "$@" 2>/dev/null || "$@"

