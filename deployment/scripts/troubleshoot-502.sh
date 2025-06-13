#!/bin/bash
# DigitalOcean Droplet Troubleshooting Script
# Run this on your DO droplet to diagnose 502 errors

echo "🔍 DigitalOcean Droplet Troubleshooting for 502 Errors"
echo "=================================================="
echo

# Function to check and report status
check_status() {
    local service=$1
    local command=$2
    echo -n "🔍 Checking $service... "
    if eval "$command" &>/dev/null; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FAILED"
        return 1
    fi
}

# 1. Check if nginx is installed and running
echo "1️⃣  NGINX STATUS"
echo "----------------"
check_status "nginx installation" "which nginx"
check_status "nginx service" "systemctl is-active nginx"
if ! systemctl is-active nginx &>/dev/null; then
    echo "   🔧 Try: sudo systemctl start nginx"
fi

# Test nginx configuration
echo -n "🔍 Testing nginx config... "
if nginx -t &>/dev/null; then
    echo "✅ Valid"
else
    echo "❌ Invalid"
    echo "   📋 Nginx config errors:"
    nginx -t 2>&1 | sed 's/^/      /'
fi
echo

# 2. Check FastAPI application
echo "2️⃣  FASTAPI APPLICATION"
echo "----------------------"
check_status "uvicorn process" "pgrep -f 'uvicorn.*app:app'"
check_status "port 8000 listening" "ss -tln | grep ':8000 ' | grep LISTEN"
check_status "application health (direct)" "curl -sf http://127.0.0.1:8000/health"

if ! pgrep -f "uvicorn.*app:app" &>/dev/null; then
    echo "   🔧 Application not running. Check logs:"
    echo "   📋 Recent application attempts:"
    journalctl -u uvicorn --no-pager -n 10 2>/dev/null || echo "      No systemd logs found"
    echo "   📋 Check if app directory exists:"
    ls -la /opt/photogrammetry/ 2>/dev/null | head -5 || echo "      /opt/photogrammetry/ not found"
fi
echo

# 3. Check reverse proxy
echo "3️⃣  REVERSE PROXY"
echo "----------------"
check_status "port 80 listening" "ss -tln | grep ':80 ' | grep LISTEN"
check_status "nginx config exists" "test -f /etc/nginx/sites-enabled/photogrammetry"
check_status "reverse proxy working" "curl -sf http://localhost/health"
echo

# 4. Check networking and firewall
echo "4️⃣  NETWORKING & FIREWALL"
echo "-------------------------"
check_status "ufw firewall" "ufw status | grep -E '(80|443).*ALLOW'"
check_status "iptables rules" "iptables -L INPUT | grep -E '(80|443|ACCEPT)'"

echo "📋 Active ports:"
ss -tln | grep -E ':(80|443|8000) ' | sed 's/^/   /'
echo

# 5. Check system resources
echo "5️⃣  SYSTEM RESOURCES"
echo "-------------------"
echo "📊 Memory usage:"
free -h | sed 's/^/   /'
echo
echo "📊 Disk usage:"
df -h / | sed 's/^/   /'
echo
echo "📊 Load average:"
uptime | sed 's/^/   /'
echo

# 6. Check logs
echo "6️⃣  ERROR LOGS"
echo "-------------"
echo "📋 Recent nginx errors (last 10 lines):"
tail -10 /var/log/nginx/error.log 2>/dev/null | sed 's/^/   /' || echo "   No nginx error logs found"
echo
echo "📋 Recent nginx access logs (last 5 lines):"
tail -5 /var/log/nginx/access.log 2>/dev/null | sed 's/^/   /' || echo "   No nginx access logs found"
echo
echo "📋 System logs related to nginx/python:"
journalctl -u nginx --no-pager -n 5 2>/dev/null | sed 's/^/   /' || echo "   No nginx systemd logs"
echo

# 7. Suggested fixes
echo "7️⃣  SUGGESTED FIXES"
echo "------------------"
echo "If application is not running:"
echo "   cd /opt/photogrammetry"
echo "   python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 &"
echo
echo "If nginx is not configured:"
echo "   sudo cp deployment/nginx/http.conf /etc/nginx/sites-available/photogrammetry"
echo "   sudo ln -sf /etc/nginx/sites-available/photogrammetry /etc/nginx/sites-enabled/"
echo "   sudo nginx -t && sudo systemctl reload nginx"
echo
echo "If firewall is blocking:"
echo "   sudo ufw allow 80"
echo "   sudo ufw allow 443"
echo
echo "To restart everything:"
echo "   sudo systemctl restart nginx"
echo "   killall uvicorn 2>/dev/null || true"
echo "   cd /opt/photogrammetry && python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 &"
echo
echo "🔍 For detailed status: ./deployment/scripts/check-deployment.sh"
