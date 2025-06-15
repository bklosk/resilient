#!/bin/bash

# Emergency fix for HTTPS deployment
echo "🚨 Emergency HTTPS Fix"
echo "====================="

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose -f docker-compose.production.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.ip.yml down --remove-orphans 2>/dev/null || true
docker stop resilience-api photogrammetry-caddy 2>/dev/null || true
docker rm resilience-api photogrammetry-caddy 2>/dev/null || true

# Clear certificate cache that might be corrupted
echo "🧹 Clearing certificate cache..."
docker volume rm $(docker volume ls -q | grep caddy) 2>/dev/null || true

# Try production deployment first
echo "🚀 Attempting production deployment..."
if docker compose -f docker-compose.production.yml up -d --build; then
    echo "✅ Production deployment started"
    echo "⏳ Waiting for services..."
    sleep 30
    
    # Test HTTPS
    if curl -f -s https://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
        echo "✅ HTTPS working with production deployment!"
        exit 0
    elif curl -f -s http://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
        echo "⚠️ HTTP working, HTTPS certificates may need time"
        echo "💡 Check https://api.climateriskplan.com in 5 minutes"
        exit 0
    fi
fi

echo "⚠️ Production deployment failed, trying IP fallback..."
docker compose -f docker-compose.production.yml down 2>/dev/null || true

# Fallback to IP deployment
echo "🔄 Using IP deployment fallback..."
if docker compose -f docker-compose.ip.yml up -d --build; then
    echo "✅ IP deployment started"
    echo "⏳ Waiting for services..."
    sleep 20
    
    # Test HTTP
    if curl -f -s http://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
        echo "✅ HTTP working with IP deployment!"
        echo "⚠️ HTTPS uses self-signed certificate (browser will show warning)"
        exit 0
    fi
fi

echo "❌ Both deployments failed. Check logs:"
docker compose -f docker-compose.production.yml logs --tail=20 2>/dev/null || true
docker compose -f docker-compose.ip.yml logs --tail=20 2>/dev/null || true
