#!/bin/bash

echo "🎯 Testing Simple RabbitMQ-Based ZXing Architecture"
echo "=================================================="

# Build the Docker images
echo "📦 Building Docker images..."
docker build -f Dockerfile.simple-worker -t simple-worker:latest .
docker build -f Dockerfile.simple-client -t simple-client:latest .

# Start RabbitMQ and worker
echo "🚀 Starting RabbitMQ and simple worker..."
docker-compose -f docker-compose-simple.yml up -d rabbitmq simple-worker

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check RabbitMQ management UI
echo "🔍 RabbitMQ Management UI available at: http://localhost:15672"
echo "   Username: admin"
echo "   Password: admin123"

# Run the test client
echo "🧪 Running simple test client..."
docker-compose -f docker-compose-simple.yml up simple-test-client

# Show queue status
echo "📊 Queue Status:"
docker exec simple-rabbitmq rabbitmqctl list_queues name messages consumers

# Clean up
echo "🧹 Cleaning up..."
docker-compose -f docker-compose-simple.yml down

echo "✅ Simple RabbitMQ test completed!"
