# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir Flask cloudpickle requests

# Create a workspace directory for user files
RUN mkdir /workspace

# Copy the API script into the image
# This script will be created in a later step.
# For now, the Dockerfile will expect it to be in the same directory during build.
COPY sandbox_api.py .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "sandbox_api.py"]
