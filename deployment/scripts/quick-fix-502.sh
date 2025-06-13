#!/bin/bash
# Quick Fix Script for 502 Errors on DigitalOcean Droplet
# Run this script to quickly resolve common deployment issues

set -e

echo "🚨 Quick Fix for 502 Errors"
echo "============================"

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "📁 Changing to application directory..."
    cd /opt/photogrammetry || {
        echo "❌ Application directory not found. Has deployment run?"
        exit 1
    }
fi

# 1. Stop everything first
echo "🛑 Stopping existing services..."
sudo systemctl stop nginx 2>/dev/null || true
killall -q uvicorn 2>/dev/null || true

# 2. Install/update dependencies if needed
echo "📦 Checking dependencies..."
if ! which nginx >/dev/null; then
    echo "   Installing nginx..."
    sudo apt update && sudo apt install -y nginx
fi

if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "   Installing Python dependencies..."
    python3 -m pip install -r requirements.txt --user
fi

# 3. Configure nginx
echo "⚙️  Configuring nginx..."
if [ -f "deployment/nginx/http.conf" ]; then
    sudo cp deployment/nginx/http.conf /etc/nginx/sites-available/photogrammetry
else
    echo "❌ Nginx config not found, creating basic config..."
    sudo tee /etc/nginx/sites-available/photogrammetry > /dev/null << 'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name _;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
EOF
fi

# Enable the site
sudo ln -sf /etc/nginx/sites-available/photogrammetry /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx config
echo "🧪 Testing nginx configuration..."
if ! sudo nginx -t; then
    echo "❌ Nginx configuration invalid"
    exit 1
fi

# 4. Configure firewall
echo "🔥 Configuring firewall..."
if sudo ufw status | grep -q "Status: active"; then
    sudo ufw allow 80/tcp >/dev/null 2>&1 || true
    sudo ufw allow 443/tcp >/dev/null 2>&1 || true
fi

# 5. Start nginx
echo "🌐 Starting nginx..."
sudo systemctl start nginx
sudo systemctl enable nginx

# 6. Start FastAPI application
echo "🐍 Starting FastAPI application..."
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 &
APP_PID=$!

# 7. Wait and test
echo "⏳ Waiting for services to start..."
sleep 5

# Test application directly
echo "🔍 Testing application..."
if curl -f http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "✅ FastAPI application responding"
else
    echo "❌ FastAPI application not responding"
    echo "📋 Checking if process is running..."
    if kill -0 $APP_PID 2>/dev/null; then
        echo "   Process is running (PID: $APP_PID), but not responding to requests"
        echo "   This might be a dependency or import error"
        echo "   Try running manually: python3 -m uvicorn app:app --host 127.0.0.1 --port 8000"
    else
        echo "   Process died. Check for Python errors:"
        python3 -c "import app" 2>&1 || echo "   Import failed - check dependencies"
    fi
    exit 1
fi

# Test reverse proxy
echo "🔍 Testing reverse proxy..."
if curl -f http://localhost/health >/dev/null 2>&1; then
    echo "✅ Reverse proxy working"
else
    echo "❌ Reverse proxy not working"
    echo "📋 Nginx status: $(sudo systemctl is-active nginx)"
    echo "📋 Recent nginx errors:"
    sudo tail -5 /var/log/nginx/error.log 2>/dev/null || echo "   No error logs"
    exit 1
fi

# Get external IP for final test
EXTERNAL_IP=$(curl -s http://ifconfig.me 2>/dev/null || echo "unknown")

echo ""
echo "🎉 SUCCESS! Services are running"
echo "================================"
echo "📊 Status:"
echo "   • FastAPI Application PID: $APP_PID"
echo "   • Nginx Status: $(sudo systemctl is-active nginx)"
echo "   • Local Health Check: ✅"
echo "   • Reverse Proxy: ✅"
echo ""
echo "🌐 Access your API:"
echo "   • Health Check: http://$EXTERNAL_IP/health"
echo "   • API Docs: http://$EXTERNAL_IP/docs"
echo "   • Full API: http://$EXTERNAL_IP/"
echo ""
echo "🔍 Monitor with: ./deployment/scripts/check-deployment.sh"
echo "🔧 Troubleshoot with: ./deployment/scripts/troubleshoot-502.sh"
