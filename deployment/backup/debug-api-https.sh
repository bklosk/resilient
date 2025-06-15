#!/bin/bash

echo "🔍 HTTPS Debug for api.climateriskplan.com"
echo "=========================================="

# Check current containers
echo "📦 Current containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "🌐 DNS Check:"
echo "api.climateriskplan.com resolves to: $(dig +short api.climateriskplan.com)"
echo "Server IP: $(curl -s https://ipinfo.io/ip)"

echo
echo "🔗 Connectivity Tests:"
echo "HTTP (should work):"
curl -I -s http://api.climateriskplan.com/ | head -3

echo
echo "HTTPS (checking certificate):"
curl -I -s https://api.climateriskplan.com/ 2>&1 | head -5

echo
echo "📝 Recent Caddy logs (last 30 lines):"
docker logs photogrammetry-caddy --tail 30

echo
echo "🔐 Certificate Information:"
docker exec photogrammetry-caddy caddy list-certificates 2>/dev/null || echo "No certificates found or Caddy not running"

echo
echo "📋 Current Caddyfile in container:"
docker exec photogrammetry-caddy cat /etc/caddy/Caddyfile

echo
echo "💡 Next Steps:"
echo "=============="
echo "1. If DNS doesn't match server IP, update A record"
echo "2. If containers aren't running, deploy with: ./deploy-domain.sh"
echo "3. If certificate acquisition failed, check logs above"
echo "4. Wait a few minutes after deployment for certificate to be issued"
