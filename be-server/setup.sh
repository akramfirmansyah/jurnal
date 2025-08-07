# remove the old Docker container if it exists
docker rm -f flask-container || true

# remove the old Docker image if it exists
docker rmi -f flask-app || true

# Build the Docker image
docker build -t flask-app .

# Run the Docker container
docker run -d -P --name flask-container flask-app

# Print a message indicating that the setup is complete
echo "Flask application is running in a Docker container on port 5000."
