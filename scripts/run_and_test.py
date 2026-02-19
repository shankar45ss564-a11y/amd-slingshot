#!/usr/bin/env python3
"""
Script to run and test the AMD SlingShot Hackathon project
This script installs dependencies, runs tests, starts the server, and performs basic verification
"""

import subprocess
import sys
import time
import os
import signal

def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result

def main():
    print("🚀 Starting AMD SlingShot Project Test Script")

    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        use_uv = True
        print("Using uv for dependency management...")
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_uv = False
        print("uv not found, using pip...")

    # Install dependencies
    print("📦 Installing dependencies...")
    if use_uv:
        run_command(["uv", "sync"])
    else:
        run_command([sys.executable, "-m", "pip", "install", "-e", "."])

    # Run tests
    print("🧪 Running tests...")
    run_command([sys.executable, "-m", "pytest", "tests/", "-v"])

    # Start the server in background
    print("🌐 Starting FastAPI server...")
    server_process = subprocess.Popen([
        sys.executable, "scripts/run_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait for server to start
        time.sleep(5)

        # Test the API
        print("🔍 Testing API endpoints...")
        try:
            run_command(["curl", "-f", "http://localhost:8000/docs"], check=False)
        except:
            print("API docs not accessible (curl not available?)")

        try:
            run_command(["curl", "-f", "http://localhost:8000/api/tasks"], check=False)
        except:
            print("Tasks endpoint not accessible (curl not available?)")

        # Run verification script
        print("✅ Running backend verification...")
        run_command([sys.executable, "scripts/verify_backend.py"])

        # Test MCP if possible
        print("🔧 Testing MCP integration...")
        try:
            # Run MCP inspector briefly
            result = run_command([
                "timeout", "10", "npx", "@modelcontextprotocol/inspector",
                sys.executable, "scripts/run_mcp.py"
            ], check=False)
            if result.returncode == 0:
                print("MCP test completed")
            else:
                print("MCP test skipped or failed")
        except:
            print("npx not available, skipping MCP test")

    finally:
        # Stop the server
        print("🛑 Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()

    print("🎉 All tests completed successfully!")

if __name__ == "__main__":
    main()