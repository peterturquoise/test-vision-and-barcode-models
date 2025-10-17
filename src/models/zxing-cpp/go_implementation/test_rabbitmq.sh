#!/bin/bash

echo "🐰 Testing RabbitMQ-Based ZXing Architecture"
echo "============================================="

# Build the Docker images
echo "📦 Building Docker images..."
docker build -f Dockerfile.rabbitmq-worker -t zxing-rabbitmq-worker:latest .
docker build -f Dockerfile.rabbitmq-client -t zxing-rabbitmq-client:latest .

# Start RabbitMQ and worker
echo "🚀 Starting RabbitMQ and worker..."
docker-compose -f docker-compose-rabbitmq.yml up -d rabbitmq zxing-worker

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check RabbitMQ management UI
echo "🔍 RabbitMQ Management UI available at: http://localhost:15672"
echo "   Username: admin"
echo "   Password: admin123"

# Run the test client
echo "🧪 Running test client..."
docker-compose -f docker-compose-rabbitmq.yml up test-client

# Show queue status
echo "📊 Queue Status:"
docker exec zxing-rabbitmq rabbitmqctl list_queues name messages consumers

# Clean up
echo "🧹 Cleaning up..."
docker-compose -f docker-compose-rabbitmq.yml down

echo "✅ RabbitMQ test completed!"
