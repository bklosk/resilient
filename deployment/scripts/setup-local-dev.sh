#!/bin/bash
# Local Development Setup Script
# Use this to set up the photogrammetry API with nginx reverse proxy locally

set -e

echo "🚀 Setting up Photogrammetry API for local development..."

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "📦 Installing nginx..."
    sudo apt update
    sudo apt install -y nginx
fi

# Stop any existing services
echo "🛑 Stopping existing services..."
sudo service nginx stop 2>/dev/null || true
killall -q uvicorn 2>/dev/null || true

# Install Python dependencies
echo "📚 Installing Python dependencies..."
python3 -m pip install -r requirements.txt --user

# Configure nginx
echo "⚙️  Configuring nginx..."
sudo cp deployment/nginx/http.conf /etc/nginx/sites-available/photogrammetry
sudo ln -sf /etc/nginx/sites-available/photogrammetry /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
echo "🧪 Testing nginx configuration..."
sudo nginx -t

# Start nginx
echo "🌐 Starting nginx..."
sudo service nginx start

# Start FastAPI application
echo "🐍 Starting FastAPI application..."
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 &
APP_PID=$!

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 3

# Test the setup
echo "🔍 Testing the setup..."
if curl -f http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ FastAPI application is responding"
else
    echo "❌ FastAPI application is not responding"
    exit 1
fi

if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Nginx reverse proxy is working"
else
    echo "❌ Nginx reverse proxy is not working"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Service Status:"
echo "  • FastAPI Application: http://127.0.0.1:8000"
echo "  • API via Nginx: http://localhost"
echo "  • API Documentation: http://localhost/docs"
echo "  • Health Check: http://localhost/health"
echo ""
echo "📊 To monitor status: ./deployment/scripts/check-deployment.sh"
echo "🛑 To stop: killall uvicorn && sudo service nginx stop"
echo ""
echo "🔧 Application PID: $APP_PID"
