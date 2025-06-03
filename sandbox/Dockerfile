# Base image
FROM python:3.10-slim

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir Flask cloudpickle requests

# Create a workspace directory for user files
RUN mkdir /workspace && chown appuser:appuser /workspace

# Copy the API script into the image
COPY sandbox_api.py .

# Change ownership of the application directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Limit container resources (example values, adjust as needed)
# Note: These limits need to be set when running the container, not in the Dockerfile.
# Example: docker run --memory="256m" --cpus="0.5" myimage
# For documentation purposes, including these comments here for clarity.

# Command to run the application
CMD ["python", "sandbox_api.py"]

