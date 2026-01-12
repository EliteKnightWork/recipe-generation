# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create models directory
RUN mkdir -p /app/models

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run entrypoint script when the container launches
ENTRYPOINT ["./entrypoint.sh"]
