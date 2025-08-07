# remove the old Docker container if it exists
docker rm -f flask-container || true

# Build the Docker image
docker build -t flask-app .

# Run the Docker container
docker run -d -p 5000:5000 flask-app --name flask-container
