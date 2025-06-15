#!/bin/bash

# Manual deployment script with debugging
# Run this on your server to diagnose and fix the deployment

set -e

echo "🚀 Manual Deployment with Debugging"
echo "===================================="

# Change to deployment directory
cd /opt/photogrammetry/deployment || {
    echo "❌ Could not find /opt/photogrammetry/deployment"
    echo "Make sure the application is extracted to /opt/photogrammetry"
    exit 1
}

# Determine compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ No Docker Compose found!"
    exit 1
fi

echo "Using: $COMPOSE_CMD"

# Function to check port
check_port() {
    local port=$1
    if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        echo "⚠️ Port $port is in use:"
        netstat -tuln 2>/dev/null | grep ":$port " || ss -tuln 2>/dev/null | grep ":$port "
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

# Stop any existing containers
echo "🛑 Stopping existing containers..."
$COMPOSE_CMD -f docker-compose.ip.yml down --remove-orphans 2>/dev/null || true
$COMPOSE_CMD -f docker-compose.dev.yml down --remove-orphans 2>/dev/null || true
docker stop resilience-api photogrammetry-caddy 2>/dev/null || true
docker rm resilience-api photogrammetry-caddy 2>/dev/null || true

# Check ports
echo -e "\n🔍 Checking ports..."
PORT_80_OK=true
PORT_443_OK=true

if ! check_port 80; then
    PORT_80_OK=false
    echo "Trying to free port 80..."
    systemctl stop nginx 2>/dev/null || true
    systemctl stop apache2 2>/dev/null || true
    fuser -k 80/tcp 2>/dev/null || true
    sleep 3
    if check_port 80; then
        PORT_80_OK=true
    fi
fi

if ! check_port 443; then
    PORT_443_OK=false
    echo "Trying to free port 443..."
    fuser -k 443/tcp 2>/dev/null || true
    sleep 3
    if check_port 443; then
        PORT_443_OK=true
    fi
fi

# Validate Caddyfile
echo -e "\n📝 Validating Caddyfile.ip..."
if docker run --rm -v "$(pwd)/Caddyfile.ip:/etc/caddy/Caddyfile:ro" caddy:latest caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile; then
    echo "✅ Caddyfile.ip is valid"
else
    echo "❌ Caddyfile.ip has errors - falling back to dev config"
    PORT_80_OK=false
    PORT_443_OK=false
fi

# Decide deployment strategy
if [ "$PORT_80_OK" = true ] && [ "$PORT_443_OK" = true ]; then
    echo -e "\n🚀 Deploying with HTTPS (docker-compose.ip.yml)..."
    COMPOSE_FILE="docker-compose.ip.yml"
    DEPLOYMENT_TYPE="production"
else
    echo -e "\n🚀 Deploying with HTTP fallback (docker-compose.dev.yml)..."
    COMPOSE_FILE="docker-compose.dev.yml"
    DEPLOYMENT_TYPE="development"
fi

# Deploy
echo "Starting containers with $COMPOSE_FILE..."
if $COMPOSE_CMD -f $COMPOSE_FILE up -d --build; then
    echo "✅ Containers started successfully"
else
    echo "❌ Container startup failed. Showing logs:"
    $COMPOSE_CMD -f $COMPOSE_FILE logs --tail=50
    exit 1
fi

# Wait for startup
echo -e "\n⏳ Waiting for services to start..."
sleep 15

# Check container status
echo -e "\n📊 Container status:"
$COMPOSE_CMD -f $COMPOSE_FILE ps

# Test the deployment
echo -e "\n🔍 Testing deployment..."
if [ "$DEPLOYMENT_TYPE" = "production" ]; then
    # Test HTTPS
    if curl -f -k --max-time 30 https://localhost/health >/dev/null 2>&1; then
        echo "✅ HTTPS deployment successful!"
        echo "🌐 Your API is available at: https://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')"
        echo "🌐 Also available at: http://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}') (redirects to HTTPS)"
    elif curl -f --max-time 30 http://localhost/health >/dev/null 2>&1; then
        echo "✅ HTTP deployment working (HTTPS may need more time)"
        echo "🌐 Your API is available at: http://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')"
        echo "💡 Try https://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}') in a few minutes"
    else
        echo "❌ Deployment test failed"
        echo "📋 Checking logs..."
        $COMPOSE_CMD -f $COMPOSE_FILE logs --tail=30 caddy
    fi
else
    # Test HTTP on 8080
    if curl -f --max-time 30 http://localhost:8080/health >/dev/null 2>&1; then
        echo "✅ HTTP deployment successful!"
        echo "🌐 Your API is available at: http://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}'):8080"
    else
        echo "❌ Deployment test failed"
        echo "📋 Checking logs..."
        $COMPOSE_CMD -f $COMPOSE_FILE logs --tail=30
    fi
fi

echo -e "\n📝 To view logs in real-time:"
echo "   $COMPOSE_CMD -f $COMPOSE_FILE logs -f"
echo -e "\n📝 To stop the deployment:"
echo "   $COMPOSE_CMD -f $COMPOSE_FILE down"

echo -e "\n✅ Deployment completed!"
