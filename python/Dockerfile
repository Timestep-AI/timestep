FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and README
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY timestep/ ./timestep/

# Expose port
EXPOSE 9999

# Run the server
CMD ["uv", "run", "python", "-m", "timestep.api.server"]

