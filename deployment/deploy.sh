#!/bin/bash

# Production Deployment Script for api.climateriskplan.com
# This replaces all the other deployment scripts

set -e

echo "🚀 Deploying Photogrammetry API to Production"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "docker-compose.production.yml" ]; then
    echo "❌ Error: Must run from deployment directory"
    echo "   cd /path/to/photogrammetry/deployment && ./deploy.sh"
    exit 1
fi

# Check DNS resolution
echo "🔍 Checking DNS resolution..."
DOMAIN_IP=$(dig +short api.climateriskplan.com)
SERVER_IP=$(curl -s https://ipinfo.io/ip)

echo "   Domain resolves to: $DOMAIN_IP"
echo "   Server IP is: $SERVER_IP"

if [ "$DOMAIN_IP" != "$SERVER_IP" ]; then
    echo "⚠️  WARNING: Domain doesn't point to this server!"
    echo "   Please update DNS A record for api.climateriskplan.com to point to $SERVER_IP"
    echo "   HTTPS certificate will fail if DNS is incorrect"
    echo
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
fi

# Stop existing deployments
echo
echo "🛑 Stopping existing deployments..."
docker compose -f docker-compose.production.yml down --remove-orphans 2>/dev/null || true

# Clean up any orphaned containers
echo "🧹 Cleaning up orphaned containers..."
docker container prune -f || true

# Start production deployment
echo
echo "🚀 Starting production deployment..."
docker compose -f docker-compose.production.yml up -d --build

# Wait for services to start
echo
echo "⏳ Waiting for services to start..."
sleep 15

# Check deployment status
echo
echo "📊 Deployment Status:"
docker compose -f docker-compose.production.yml ps

# Test the deployment
echo
echo "🔍 Testing deployment..."
echo "Testing HTTPS:"
if curl -f -s -I https://api.climateriskplan.com/health | head -1; then
    echo "✅ HTTPS is working!"
else
    echo "⚠️  HTTPS not responding yet (certificate may still be obtaining)"
fi

echo
echo "Testing HTTP (should redirect to HTTPS):"
curl -f -s -I http://api.climateriskplan.com/health | head -1 || echo "HTTP redirect check"

# Show logs
echo
echo "📝 Recent Caddy logs:"
docker logs photogrammetry-caddy --tail 10

echo
echo "✅ Deployment completed!"
echo
echo "🌐 Your API should be available at:"
echo "   https://api.climateriskplan.com"
echo
echo "🔧 Useful commands:"
echo "   View logs: docker compose -f docker-compose.production.yml logs -f"
echo "   Stop: docker compose -f docker-compose.production.yml down"
echo "   Restart: docker compose -f docker-compose.production.yml restart"
