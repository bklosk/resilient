#!/bin/bash

echo "🔍 HTTPS Debug Script"
echo "===================="

# Check if we're running the right containers
echo
echo "📦 Current containers and their ports:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "🔗 Testing HTTPS connectivity:"

# Test HTTP first
echo "Testing HTTP (should work):"
curl -I -s http://localhost/ | head -5

echo
echo "Testing HTTPS with self-signed cert (should work with -k flag):"
curl -I -s -k https://localhost/ | head -5

echo
echo "Testing HTTPS without -k flag (will fail with self-signed):"
curl -I -s https://localhost/ 2>&1 | head -5

# Check Caddy logs for TLS/certificate issues
echo
echo "📝 Recent Caddy logs (last 20 lines):"
docker logs photogrammetry-caddy --tail 20

echo
echo "🔧 Caddy TLS certificate info:"
docker exec photogrammetry-caddy caddy list-certificates

echo
echo "💡 HTTPS Status Summary:"
echo "========================"
echo "✅ If HTTP works: Your reverse proxy is functioning"
echo "✅ If HTTPS works with -k: Self-signed certificate is working"
echo "⚠️  If HTTPS fails without -k: This is expected with self-signed certificates"
echo
echo "To fix HTTPS, you have these options:"
echo "1. Accept the self-signed certificate in your browser (click 'Advanced' → 'Proceed')"
echo "2. Use a domain name and Let's Encrypt for trusted certificates"
echo "3. Create and install a proper SSL certificate"
