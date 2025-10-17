#!/bin/bash

# Build script for Go ZXing implementation

echo "🔨 Building Go ZXing implementation..."

# Build the main application
echo "Building main application..."
go build -o zxing-go main.go

# Build the load test
echo "Building load test..."
go build -o load_test load_test.go

echo "✅ Build complete!"
echo ""
echo "To run the application:"
echo "  ./zxing-go"
echo ""
echo "To run load test:"
echo "  ./load_test"
echo ""
echo "To build Docker image:"
echo "  docker build -t zxing-go ."
