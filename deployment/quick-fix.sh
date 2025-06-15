#!/bin/bash

# Quick fix script for HTTPS certificate issues
# Run this on the production server

echo "🔧 Quick HTTPS Fix for api.climateriskplan.com"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "docker-compose.production.yml" ]; then
    echo "❌ Error: Must run from deployment directory"
    exit 1
fi

echo "📊 Current container status:"
docker compose -f docker-compose.production.yml ps

echo
echo "🔄 Restarting Caddy to regenerate certificates..."
docker compose -f docker-compose.production.yml restart caddy

echo "⏳ Waiting 30 seconds for Caddy to restart..."
sleep 30

echo
echo "📝 Checking Caddy logs for certificate generation:"
docker logs photogrammetry-caddy --tail 20

echo
echo "🧪 Testing HTTPS after restart:"
if curl -f -s -I https://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
    echo "✅ HTTPS is now working!"
else
    echo "❌ HTTPS still not working. Checking certificate details..."
    echo
    echo "Certificate check:"
    timeout 10 openssl s_client -connect api.climateriskplan.com:443 -servername api.climateriskplan.com </dev/null 2>&1 | grep -E "(CONNECTED|Certificate chain|Verify return code)" || echo "Certificate connection failed"
    
    echo
    echo "🔧 Try these additional steps:"
    echo "1. Check DNS: dig api.climateriskplan.com"
    echo "2. Full restart: docker compose -f docker-compose.production.yml down && docker compose -f docker-compose.production.yml up -d"
    echo "3. Clear certificate cache: docker volume rm \$(docker volume ls -q | grep caddy)"
    echo "4. Check if ports 80/443 are accessible from outside"
fi

echo
echo "✅ Quick fix attempt completed!"
