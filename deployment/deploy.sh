#!/bin/bash
# Minimal Production Deployment Script
set -e

echo "🚀 Deploying to api.climateriskplan.com"

# Validate environment
[ ! -f "docker-compose.yml" ] && echo "❌ Run from deployment directory" && exit 1

# DNS check (non-blocking)
DOMAIN_IP=$(dig +short api.climateriskplan.com 2>/dev/null || echo "unknown")
SERVER_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || echo "unknown")
echo "📍 DNS: $DOMAIN_IP → Server: $SERVER_IP"

# Deploy
echo "🛑 Stopping existing containers..."
docker compose down --remove-orphans 2>/dev/null || true
docker container prune -f >/dev/null 2>&1 || true

echo "🚀 Starting deployment..."
docker compose up -d --build

# Monitor startup
echo "⏳ Monitoring startup..."
for i in {1..12}; do
    if docker compose ps --format json | jq -r '.[].Health' 2>/dev/null | grep -q "healthy"; then
        echo "✅ Services healthy"
        break
    fi
    [ $i -eq 12 ] && echo "⚠️ Startup taking longer than expected"
    echo "   Attempt $i/12..."
    sleep 5
done

# Test endpoints
echo "🔍 Testing endpoints..."
if curl -fs https://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
    echo "✅ HTTPS working: https://api.climateriskplan.com"
elif curl -fs http://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
    echo "⚠️ HTTP working, HTTPS initializing: http://api.climateriskplan.com"
else
    echo "❌ Endpoints not responding yet"
    docker compose logs --tail=10 caddy
fi

echo "📊 Status:"
docker compose ps
echo "✅ Deployment complete"
