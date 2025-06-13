#!/bin/bash
# Deployment Status Check Script
# Checks if the application and nginx are running correctly

echo "=== Photogrammetry API Deployment Status ==="
echo

# Check if nginx is running
echo "🔍 Checking nginx status..."
if pgrep nginx > /dev/null; then
    echo "✅ Nginx is running"
    if sudo nginx -t 2>/dev/null; then
        echo "✅ Nginx configuration is valid"
    else
        echo "❌ Nginx configuration has errors"
    fi
else
    echo "❌ Nginx is not running"
fi
echo

# Check if the application is running
echo "🔍 Checking application status..."
if pgrep -f "uvicorn app:app" > /dev/null; then
    echo "✅ FastAPI application is running"
    
    # Test local health endpoint
    if curl -s -f http://127.0.0.1:8000/health > /dev/null; then
        echo "✅ Application health check passed"
    else
        echo "❌ Application health check failed"
    fi
else
    echo "❌ FastAPI application is not running"
fi
echo

# Check if the reverse proxy is working
echo "🔍 Checking reverse proxy..."
if curl -s -f http://localhost/health > /dev/null; then
    echo "✅ Reverse proxy is working (HTTP)"
else
    echo "❌ Reverse proxy is not working (HTTP)"
fi

# Check HTTPS if certificates exist
if [ -f "/etc/letsencrypt/live/$(hostname -f)/fullchain.pem" ] 2>/dev/null; then
    echo "🔍 Checking HTTPS..."
    if curl -s -f https://localhost/health > /dev/null; then
        echo "✅ HTTPS is working"
    else
        echo "❌ HTTPS is not working"
    fi
else
    echo "ℹ️  HTTPS not configured (no SSL certificates found)"
fi
echo

# Show listening ports
echo "🔍 Active listening ports:"
netstat -tlnp | grep -E ':(80|443|8000) ' | sed 's/^/  /'
echo

# Show recent nginx logs
echo "📋 Recent nginx access logs (last 5 lines):"
tail -n 5 /var/log/nginx/access.log 2>/dev/null | sed 's/^/  /' || echo "  No nginx access logs found"
echo

# Show any nginx errors
echo "📋 Recent nginx errors (last 5 lines):"
tail -n 5 /var/log/nginx/error.log 2>/dev/null | sed 's/^/  /' || echo "  No nginx error logs found"
echo

echo "=== Status check complete ==="
