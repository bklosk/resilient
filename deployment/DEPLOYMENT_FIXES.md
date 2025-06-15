# Deployment Failure Analysis & Fixes

## Root Cause Analysis

### Issue 1: Wrong Deployment Configuration
**Problem**: GitHub Actions workflow was attempting to use `docker-compose.ip.yml` (IP-based deployment with self-signed certificates) instead of `docker-compose.production.yml` (domain-based deployment with Let's Encrypt).

**Evidence**: Error message showed `open /opt/photogrammetry/deployment/docker-compose.ip.yml: no such file or directory`

**Impact**: 
- Health checks timing out after 20 attempts
- Containers not starting properly
- Wrong SSL certificate configuration

### Issue 2: Health Check Dependencies
**Problem**: Health check commands in docker-compose files were using `curl` which isn't available in the containers.

**Evidence**: Empty status for both app and caddy containers in health check output

**Fix Applied**:
- App container: Changed from `curl` to Python's `urllib.request`
- Caddy container: Changed from `curl` to `wget` (available in Alpine Linux)

### Issue 3: Incorrect Health Endpoints
**Problem**: Some health checks were testing `/health` instead of `/health/ready`

**Fix Applied**: Updated all health check endpoints to use `/health/ready`

## Files Fixed

### 1. `/deployment/docker-compose.ip.yml`
- ✅ Fixed app health check to use Python instead of curl
- ✅ Fixed caddy health check to use wget instead of curl
- ✅ Updated health endpoint to `/health/ready`
- ✅ Removed inefficient `apk add curl` command

### 2. `/deployment/docker-compose.dev.yml`  
- ✅ Fixed app health check to use Python instead of curl
- ✅ Fixed caddy health check to use wget instead of curl
- ✅ Updated health endpoint to `/health/ready`

### 3. `/deployment/docker-compose.production.yml`
- ✅ Already properly configured
- ✅ Uses correct domain configuration
- ✅ Proper health checks in place

## Deployment Strategy

### For Production Domain (api.climateriskplan.com)
**Correct Method**: Use `docker-compose.production.yml`
```bash
cd /opt/photogrammetry/deployment
./deploy.sh  # Uses production.yml
# OR
./fix-production.sh  # Force production deployment
```

### For IP-based Development
**Fallback Method**: Use `docker-compose.ip.yml` (now fixed)
```bash
cd /opt/photogrammetry/deployment  
docker compose -f docker-compose.ip.yml up -d --build
```

### For Development/Testing
**Development Method**: Use `docker-compose.dev.yml` (port 8080)
```bash
cd /opt/photogrammetry/deployment
docker compose -f docker-compose.dev.yml up -d --build
```

## Monitoring Commands

```bash
# Check container status
docker compose -f docker-compose.production.yml ps

# View health check logs
docker logs resilience-api --tail 20
docker logs photogrammetry-caddy --tail 20

# Test endpoints
curl http://api.climateriskplan.com/health/ready
curl https://api.climateriskplan.com/health/ready
```

## Prevention

1. **Use correct deployment file**: Always use `docker-compose.production.yml` for domain-based production deployments
2. **Test health checks**: Ensure health check commands work in the container environment
3. **Monitor startup**: Use the monitoring scripts to track deployment progress
4. **Clear state**: When switching deployment types, clear volumes and containers first

## Quick Recovery Commands

```bash
# If deployment is stuck/failed:
cd /opt/photogrammetry/deployment

# Stop everything
docker compose -f docker-compose.ip.yml down --remove-orphans
docker compose -f docker-compose.production.yml down --remove-orphans

# Clear problematic state
docker container prune -f
docker volume rm $(docker volume ls -q | grep caddy) 2>/dev/null || true

# Start with correct configuration
./fix-production.sh
```
