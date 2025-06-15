#!/bin/bash

# Automated deployment script for GitHub Actions
# This script tries production first, then falls back gracefully

set -e

echo "🤖 Automated Deployment for api.climateriskplan.com"
echo "=================================================="

cd /opt/photogrammetry/deployment

# Function to test if deployment is working
test_deployment() {
    local scheme=$1
    local host=$2
    local port=$3
    local endpoint="${scheme}://${host}${port}/health/ready"
    
    echo "Testing: $endpoint"
    if curl -f -s --max-time 10 "$endpoint" >/dev/null 2>&1; then
        echo "✅ $endpoint is working"
        return 0
    else
        echo "❌ $endpoint failed"
        return 1
    fi
}

# Stop any existing deployments
echo "🛑 Stopping existing deployments..."
docker compose -f docker-compose.production.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.ip.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.dev.yml down --remove-orphans 2>/dev/null || true

# Clean up
echo "🧹 Cleaning up..."
docker container prune -f 2>/dev/null || true

# Try production deployment (domain-based HTTPS)
echo "🚀 Attempting production deployment..."
if docker compose -f docker-compose.production.yml up -d --build; then
    echo "⏳ Waiting for production services..."
    sleep 30
    
    if test_deployment "https" "api.climateriskplan.com" ""; then
        echo "✅ Production HTTPS deployment successful!"
        echo "🌐 API available at: https://api.climateriskplan.com"
        exit 0
    elif test_deployment "http" "api.climateriskplan.com" ""; then
        echo "⚠️ Production HTTP working, HTTPS certificates may need time"
        echo "🌐 API available at: http://api.climateriskplan.com"
        echo "💡 HTTPS should be available in 5-10 minutes"
        exit 0
    fi
    
    echo "⚠️ Production deployment not responding, trying fallback..."
    docker compose -f docker-compose.production.yml down 2>/dev/null || true
fi

# Try IP-based deployment (fallback with self-signed certs)
echo "🔄 Attempting IP deployment fallback..."
if docker compose -f docker-compose.ip.yml up -d --build; then
    echo "⏳ Waiting for IP services..."
    sleep 20
    
    # Get server IP
    SERVER_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || curl -s http://ipinfo.io/ip 2>/dev/null || echo "unknown")
    
    if test_deployment "http" "$SERVER_IP" ""; then
        echo "✅ IP deployment HTTP working!"
        echo "🌐 API available at: http://$SERVER_IP"
        echo "⚠️ HTTPS available at: https://$SERVER_IP (self-signed certificate)"
        exit 0
    fi
    
    echo "⚠️ IP deployment failed, trying development fallback..."
    docker compose -f docker-compose.ip.yml down 2>/dev/null || true
fi

# Try development deployment (port 8080)
echo "🔄 Attempting development deployment fallback..."
if docker compose -f docker-compose.dev.yml up -d --build; then
    echo "⏳ Waiting for development services..."
    sleep 15
    
    # Get server IP
    SERVER_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || curl -s http://ipinfo.io/ip 2>/dev/null || echo "unknown")
    
    if test_deployment "http" "$SERVER_IP" ":8080"; then
        echo "✅ Development deployment working!"
        echo "🌐 API available at: http://$SERVER_IP:8080"
        echo "ℹ️ This is a development fallback deployment"
        exit 0
    fi
fi

echo "❌ All deployment strategies failed!"
echo "📋 Checking logs..."
docker compose -f docker-compose.production.yml logs --tail=20 2>/dev/null || true
docker compose -f docker-compose.ip.yml logs --tail=20 2>/dev/null || true
docker compose -f docker-compose.dev.yml logs --tail=20 2>/dev/null || true
exit 1
