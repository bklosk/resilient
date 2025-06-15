#!/bin/bash

# Production Domain Deployment Fix
# Use this if the GitHub Actions is trying to use docker-compose.ip.yml instead of production

echo "🔧 Fixing deployment configuration for api.climateriskplan.com"
echo "============================================================="

# Check if we're in the right directory
if [ ! -f "docker-compose.production.yml" ]; then
    echo "❌ Error: Must run from deployment directory"
    exit 1
fi

# Stop any existing deployments that might be using wrong configs
echo "🛑 Stopping any existing deployments..."
docker compose -f docker-compose.ip.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.production.yml down --remove-orphans 2>/dev/null || true

# Remove any containers that might be stuck
docker stop resilience-api photogrammetry-caddy 2>/dev/null || true
docker rm resilience-api photogrammetry-caddy 2>/dev/null || true

# Clear any volumes that might have bad certificate data
echo "🧹 Clearing certificate cache..."
docker volume rm $(docker volume ls -q | grep caddy) 2>/dev/null || true

# Start with the correct production configuration
echo "🚀 Starting production deployment..."
docker compose -f docker-compose.production.yml up -d --build

# Monitor startup
echo "⏳ Monitoring startup..."
for i in {1..20}; do
    # Check container status
    APP_STATUS=$(docker compose -f docker-compose.production.yml ps app --format "{{.Status}}" 2>/dev/null || echo "")
    CADDY_STATUS=$(docker compose -f docker-compose.production.yml ps caddy --format "{{.Status}}" 2>/dev/null || echo "")
    
    echo "   Attempt $i/20 - app: $APP_STATUS, caddy: $CADDY_STATUS"
    
    if echo "$APP_STATUS" | grep -q "healthy" && echo "$CADDY_STATUS" | grep -q "healthy"; then
        echo "✅ Both services are healthy"
        break
    fi
    
    if [ $i -eq 20 ]; then
        echo "❌ Health check timeout after 20 attempts"
        echo "📊 Current status:"
        docker compose -f docker-compose.production.yml ps
        echo "📋 Container logs:"
        docker compose -f docker-compose.production.yml logs --tail=20 app
        docker compose -f docker-compose.production.yml logs --tail=20 caddy
        exit 1
    fi
    
    sleep 6
done

# Test the deployment
echo "🔍 Testing deployment..."
if curl -f -s https://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
    echo "✅ HTTPS deployment successful!"
    echo "🌐 API available at: https://api.climateriskplan.com"
elif curl -f -s http://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
    echo "⚠️ HTTP working, HTTPS may need more time for certificates"
    echo "🌐 API available at: http://api.climateriskplan.com"
    echo "💡 Check https://api.climateriskplan.com in a few minutes"
else
    echo "❌ Deployment test failed"
    echo "🔍 Checking logs..."
    docker compose -f docker-compose.production.yml logs --tail=30 caddy
fi

echo "✅ Production deployment fix completed!"
