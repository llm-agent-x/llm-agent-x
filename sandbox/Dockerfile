# Base image
FROM python:3.10-slim

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir poetry

# Install tree for directory listing
RUN apt-get update && apt-get install -y tree && apt-get clean

COPY pyproject.toml poetry.lock ./

RUN poetry lock

# Copy all files first
COPY . .

# Install dependencies
RUN python -m poetry config virtualenvs.create false && \
    python -m poetry install --no-interaction --no-ansi

# Install additional dependencies
RUN python3 -m pip install uvicorn python-multipart

# Create a workspace directory for user files
RUN mkdir -p /workspace && chown appuser:appuser /workspace

# Change ownership of the application directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["sh", "-c", "tree && exec python3 -m uvicorn sandbox.sandbox_api:app --host 0.0.0.0 --port 5000"]
