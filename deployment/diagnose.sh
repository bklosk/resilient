#!/bin/bash
# Minimal Diagnostic Script

echo "🔍 Diagnostic for api.climateriskplan.com"
echo "========================================"

# DNS Check
DOMAIN_IP=$(dig +short api.climateriskplan.com 2>/dev/null || echo "unknown")
SERVER_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || echo "unknown") 
echo "📍 DNS: $DOMAIN_IP → Server: $SERVER_IP"
[ "$DOMAIN_IP" = "$SERVER_IP" ] && echo "✅ DNS correct" || echo "⚠️ DNS mismatch"

echo

# Container Status
echo "🐳 Containers:"
if command -v docker &> /dev/null; then
    docker compose ps 2>/dev/null || echo "❌ Docker Compose not available"
else
    echo "❌ Docker not available"
fi

echo

# Connectivity Tests
echo "🔗 Connectivity:"
if curl -fs https://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
    echo "✅ HTTPS: https://api.climateriskplan.com"
elif curl -fs http://api.climateriskplan.com/health/ready >/dev/null 2>&1; then
    echo "⚠️ HTTP only: http://api.climateriskplan.com"
    echo "💡 HTTPS may still be initializing"
else
    echo "❌ No response from api.climateriskplan.com"
fi

# Certificate Info
echo
echo "🔐 Certificate:"
timeout 5 openssl s_client -connect api.climateriskplan.com:443 -servername api.climateriskplan.com </dev/null 2>/dev/null | openssl x509 -noout -dates 2>/dev/null || echo "❌ Certificate check failed"

echo
echo "💡 Common fixes:"
echo "   docker compose restart caddy    # Restart proxy"
echo "   ./deploy.sh                     # Full redeploy"
        fi
    else
        echo "   ❌ API Container: Not running"
    fi
    

