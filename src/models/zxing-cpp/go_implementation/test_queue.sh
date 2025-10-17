#!/bin/bash

echo "🚀 Testing Queue-Based ZXing Architecture"
echo "=========================================="

# Build the Docker images
echo "📦 Building Docker images..."
docker build -f Dockerfile.worker -t zxing-queue-worker:latest .
docker build -f Dockerfile.testclient -t zxing-test-client:latest .

# Start the worker in the background
echo "🔧 Starting ZXing queue worker..."
docker run -d --name zxing-worker \
  -v /Users/peterbrooke/dev/cursor/test-7-vision-models/test_images:/app/test_images:ro \
  zxing-queue-worker:latest

# Wait a moment for the worker to start
echo "⏳ Waiting for worker to start..."
sleep 3

# Run the test client
echo "🧪 Running test client..."
docker run --rm --name zxing-test-client \
  --link zxing-worker:worker \
  -v /Users/peterbrooke/dev/cursor/test-7-vision-models/test_images:/app/test_images:ro \
  zxing-test-client:latest \
  /app/test_images/Complete\ Labels/3_J18CBEP8CCN070812400095N.jpeg 20s

# Clean up
echo "🧹 Cleaning up..."
docker stop zxing-worker
docker rm zxing-worker

echo "✅ Test completed!"
