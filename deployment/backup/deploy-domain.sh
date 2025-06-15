#!/bin/bash

echo "🚀 Deploying api.climateriskplan.com with HTTPS"
echo "==============================================="

# Check if domain points to this server
echo "🔍 Checking DNS resolution for api.climateriskplan.com..."
DOMAIN_IP=$(dig +short api.climateriskplan.com)
SERVER_IP=$(curl -s https://ipinfo.io/ip)

echo "Domain resolves to: $DOMAIN_IP"
echo "Server IP is: $SERVER_IP"

if [ "$DOMAIN_IP" != "$SERVER_IP" ]; then
    echo "⚠️  WARNING: Domain doesn't point to this server!"
    echo "   Make sure api.climateriskplan.com A record points to $SERVER_IP"
    echo "   Let's Encrypt certificate will fail if DNS is incorrect"
    echo
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
fi

# Stop current deployment
echo
echo "🛑 Stopping current deployment..."
docker compose down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.ip.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.dev.yml down --remove-orphans 2>/dev/null || true

# Start domain deployment
echo
echo "🚀 Starting domain deployment..."
docker compose -f docker-compose.domain.yml up -d --build

# Wait for services
echo
echo "⏳ Waiting for services to start..."
sleep 10

# Check status
echo
echo "📊 Deployment status:"
docker compose -f docker-compose.domain.yml ps

echo
echo "📝 Checking Caddy logs for certificate acquisition:"
sleep 5
docker logs photogrammetry-caddy --tail 20

echo
echo "✅ Deployment completed!"
echo
echo "🌐 Your app should be available at:"
echo "   https://api.climateriskplan.com"
echo "   http://api.climateriskplan.com (will redirect to HTTPS)"
echo
echo "🔧 To view logs:"
echo "   docker compose -f docker-compose.domain.yml logs -f"
echo
echo "🛑 To stop:"
echo "   docker compose -f docker-compose.domain.yml down"
