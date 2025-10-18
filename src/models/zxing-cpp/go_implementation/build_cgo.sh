#!/bin/bash

echo "🚀 Building C++ ZXing CGO Memory Worker"
echo "========================================"

# Set environment variables for CGO
export CGO_ENABLED=1
export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH}"
export CGO_CPPFLAGS="-I/usr/include/opencv4"
export CGO_LDFLAGS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_objdetect -lZXing"

# Check if OpenCV is installed
echo "📋 Checking OpenCV installation..."
if ! pkg-config --exists opencv4; then
    echo "❌ OpenCV4 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y libopencv-dev libopencv-contrib-dev
else
    echo "✅ OpenCV4 found"
fi

# Check if ZXing-CPP is installed
echo "📋 Checking ZXing-CPP installation..."
if ! ldconfig -p | grep -q libZXing; then
    echo "❌ ZXing-CPP not found. Installing..."
    # Install ZXing-CPP
    git clone --depth 1 https://github.com/nu-book/zxing-cpp.git /tmp/zxing-cpp
    cd /tmp/zxing-cpp
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=ON -DBUILD_BENCHMARKS=OFF
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd /tmp && rm -rf zxing-cpp
else
    echo "✅ ZXing-CPP found"
fi

# Initialize Go module
echo "📦 Initializing Go module..."
go mod init cpp-memory-worker-cgo 2>/dev/null || true

# Install dependencies
echo "📦 Installing Go dependencies..."
go get github.com/streadway/amqp

# Build the CGO worker
echo "🔨 Building CGO worker..."
go build -o cpp_memory_worker_cgo cpp_memory_worker_cgo.go

if [ $? -eq 0 ]; then
    echo "✅ CGO worker built successfully!"
    echo "📊 Binary size: $(du -h cpp_memory_worker_cgo | cut -f1)"
else
    echo "❌ Build failed!"
    exit 1
fi

# Build the test client
echo "🔨 Building test client..."
go build -o cpp_client_cgo cpp_client_cgo.go

if [ $? -eq 0 ]; then
    echo "✅ Test client built successfully!"
else
    echo "❌ Client build failed!"
    exit 1
fi

echo ""
echo "🎉 Build complete! Ready to test:"
echo "   Worker: ./cpp_memory_worker_cgo"
echo "   Client: ./cpp_client_cgo"
echo ""
echo "🚀 To test with Docker:"
echo "   docker-compose -f docker-compose-cpp-memory-cgo.yml up --build"
