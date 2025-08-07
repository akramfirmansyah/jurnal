# Docker file for Flask application
# This file is used to build the Docker image for the Flask application.
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["flask", "run"]