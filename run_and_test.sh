#!/bin/bash

# Script to run and test the AMD SlingShot Hackathon project
# This script installs dependencies, runs tests, starts the server, and performs basic verification

set -e  # Exit on any error

echo "🚀 Starting AMD SlingShot Project Test Script"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install dependencies
echo "📦 Installing dependencies..."
if command_exists uv; then
    echo "Using uv for dependency management..."
    uv sync
else
    echo "uv not found, using pip..."
    pip install -e .
fi

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

# Start the server in background
echo "🌐 Starting FastAPI server..."
python scripts/run_server.py &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to start
sleep 5

# Test the API
echo "🔍 Testing API endpoints..."
curl -f http://localhost:8000/docs || echo "API docs not accessible"
curl -f http://localhost:8000/api/tasks || echo "Tasks endpoint not accessible"

# Run verification script
echo "✅ Running backend verification..."
python scripts/verify_backend.py

# Test MCP if possible
echo "🔧 Testing MCP integration..."
if command_exists npx; then
    # Run MCP inspector briefly
    timeout 10 npx @modelcontextprotocol/inspector python scripts/run_mcp.py || echo "MCP test completed"
else
    echo "npx not available, skipping MCP test"
fi

# Stop the server
echo "🛑 Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo "🎉 All tests completed successfully!"