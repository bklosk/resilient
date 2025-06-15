#!/bin/bash

# HTTPS Diagnostic Script for api.climateriskplan.com
# Run this on the production server to diagnose HTTPS issues

echo "🔍 HTTPS Diagnostic for api.climateriskplan.com"
echo "==============================================="

# Check DNS resolution
echo "1. DNS Resolution:"
echo "   Domain: api.climateriskplan.com"
DOMAIN_IP=$(dig +short api.climateriskplan.com 2>/dev/null || echo "DNS lookup failed")
SERVER_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || echo "IP lookup failed")
echo "   Resolves to: $DOMAIN_IP"
echo "   Server IP: $SERVER_IP"

if [ "$DOMAIN_IP" = "$SERVER_IP" ]; then
    echo "   ✅ DNS is correctly configured"
else
    echo "   ❌ DNS mismatch - this will prevent HTTPS certificates"
fi

echo
echo "2. Port Connectivity:"
# Check if ports are accessible
if netstat -tlnp 2>/dev/null | grep -q ":80 "; then
    echo "   ✅ Port 80 is open and listening"
else
    echo "   ❌ Port 80 is not accessible"
fi

if netstat -tlnp 2>/dev/null | grep -q ":443 "; then
    echo "   ✅ Port 443 is open and listening"
else
    echo "   ❌ Port 443 is not accessible"
fi

echo
echo "3. Docker Container Status:"
if command -v docker &> /dev/null; then
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|resilience-api|caddy)"
    
    echo
    echo "4. Application Health Check:"
    if docker ps | grep -q "resilience-api"; then
        echo "   API Container: Running"
        # Test internal health endpoint
        if docker exec resilience-api curl -f -s http://localhost:8000/health/ready > /dev/null 2>&1; then
            echo "   ✅ API health check: PASSED"
        else
            echo "   ❌ API health check: FAILED"
        fi
    else
        echo "   ❌ API Container: Not running"
    fi
    
    if docker ps | grep -q "photogrammetry-caddy"; then
        echo "   Caddy Container: Running"
    else
        echo "   ❌ Caddy Container: Not running"
    fi
else
    echo "   ❌ Docker is not available"
fi

echo
echo "5. HTTPS Connectivity Test:"
echo "   Testing HTTP (port 80):"
if curl -f -s -I -m 10 http://api.climateriskplan.com/health 2>/dev/null | head -1; then
    echo "   ✅ HTTP is responding"
else
    echo "   ❌ HTTP is not responding"
fi

echo "   Testing HTTPS (port 443):"
if curl -f -s -I -m 10 https://api.climateriskplan.com/health 2>/dev/null | head -1; then
    echo "   ✅ HTTPS is responding"
else
    echo "   ❌ HTTPS is not responding"
    # Check for certificate issues
    echo "   Checking certificate status:"
    curl -s -I https://api.climateriskplan.com/health 2>&1 | grep -i "certificate\|ssl\|tls" || echo "   No specific SSL error detected"
fi

echo
echo "6. Certificate Information:"
if command -v openssl &> /dev/null; then
    echo "   Certificate details:"
    timeout 10 openssl s_client -connect api.climateriskplan.com:443 -servername api.climateriskplan.com </dev/null 2>/dev/null | openssl x509 -noout -dates -subject 2>/dev/null || echo "   ❌ Could not retrieve certificate"
else
    echo "   ❌ OpenSSL not available for certificate check"
fi

echo
echo "7. Recent Caddy Logs (if available):"
if docker ps | grep -q "photogrammetry-caddy"; then
    echo "   Last 10 log entries:"
    docker logs photogrammetry-caddy --tail 10 2>/dev/null || echo "   Could not retrieve logs"
else
    echo "   ❌ Caddy container not running"
fi

echo
echo "📋 SUMMARY:"
echo "=========="
echo "Check the items marked with ❌ above."
echo "Common issues:"
echo "1. DNS not pointing to server (update A record)"
echo "2. Firewall blocking ports 80/443"
echo "3. Containers not running (run ./deploy.sh)"
echo "4. Certificate not yet issued (wait 5-10 minutes after DNS update)"
echo "5. Previous certificate expired (restart Caddy)"

echo
echo "🔧 Quick fixes to try:"
echo "1. If DNS is wrong: Update A record and wait 5-10 minutes"
echo "2. If containers stopped: cd deployment && ./deploy.sh"
echo "3. If certificate issues: docker restart photogrammetry-caddy"
