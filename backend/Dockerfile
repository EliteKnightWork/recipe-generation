# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set venv as default Python path
ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/venv"

# Copy requirements.txt into the container
COPY requirements.txt ./

# Copy the application code into the container
COPY main.py ./
COPY download_model.py ./
COPY entrypoint.sh ./

# Copy utils module
COPY utils/ ./utils/

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create directories
RUN mkdir -p /app/models /app/venv

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run entrypoint script when the container launches
ENTRYPOINT ["./entrypoint.sh"]
