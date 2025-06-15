#!/bin/bash

# Debug script for deployment issues
set -e

echo "🔍 Deployment Debug Script"
echo "=========================="

# Check if we're in the deployment directory
if [ ! -f "docker-compose.ip.yml" ]; then
    echo "❌ Please run this script from the deployment directory"
    exit 1
fi

# Function to check port usage
check_port() {
    local port=$1
    echo "Checking port $port..."
    if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        echo "⚠️  Port $port is in use:"
        netstat -tuln 2>/dev/null | grep ":$port " || ss -tuln 2>/dev/null | grep ":$port "
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

# Check required ports
echo -e "\n🔍 Checking port availability..."
PORT_80_AVAILABLE=true
PORT_443_AVAILABLE=true

if ! check_port 80; then
    PORT_80_AVAILABLE=false
fi

if ! check_port 443; then
    PORT_443_AVAILABLE=false
fi

# Show Docker status
echo -e "\n🐳 Docker Status:"
if command -v docker &> /dev/null; then
    docker --version
    if docker info &> /dev/null; then
        echo "✅ Docker is running"
    else
        echo "❌ Docker is not running or not accessible"
        exit 1
    fi
else
    echo "❌ Docker is not installed"
    exit 1
fi

# Check Docker Compose
echo -e "\n🔧 Docker Compose Status:"
if docker compose version &> /dev/null; then
    echo "✅ Docker Compose v2 available"
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose v1 available"
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Docker Compose not found"
    exit 1
fi

# Show current containers
echo -e "\n📦 Current containers:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test Caddyfile syntax
echo -e "\n📝 Testing Caddyfile syntax:"
if docker run --rm -v "$(pwd)/Caddyfile.ip:/etc/caddy/Caddyfile:ro" caddy:latest caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile; then
    echo "✅ Caddyfile.ip syntax is valid"
else
    echo "❌ Caddyfile.ip has syntax errors"
fi

# Provide recommendations
echo -e "\n💡 Recommendations:"

if [ "$PORT_80_AVAILABLE" = false ] || [ "$PORT_443_AVAILABLE" = false ]; then
    echo "⚠️  Standard ports (80/443) are in use. Options:"
    echo "   1. Stop the service using these ports"
    echo "   2. Use docker-compose.dev.yml instead (uses port 8080)"
    echo "   3. Kill processes using these ports:"
    if [ "$PORT_80_AVAILABLE" = false ]; then
        echo "      sudo fuser -k 80/tcp"
    fi
    if [ "$PORT_443_AVAILABLE" = false ]; then
        echo "      sudo fuser -k 443/tcp"
    fi
else
    echo "✅ Ports 80 and 443 are available for IP deployment"
fi

# Show deployment commands
echo -e "\n🚀 Deployment Commands:"
echo "For IP deployment (HTTPS on 80/443):"
echo "  $COMPOSE_CMD -f docker-compose.ip.yml up -d --build"
echo ""
echo "For dev deployment (HTTP on 8080):"
echo "  $COMPOSE_CMD -f docker-compose.dev.yml up -d --build"
echo ""
echo "To view logs:"
echo "  $COMPOSE_CMD -f docker-compose.ip.yml logs -f"
echo ""
echo "To stop and clean up:"
echo "  $COMPOSE_CMD -f docker-compose.ip.yml down --remove-orphans"

echo -e "\n📊 Debug completed!"
