# Deployment Guide

This directory contains the simplified deployment configuration for the Photogrammetry API.

## Current Status (June 15, 2025)
✅ **HTTP API**: Working at http://api.climateriskplan.com  
❌ **HTTPS**: Certificate issues detected  
✅ **Backend**: API responding normally  
✅ **DNS**: Domain resolving correctly  
⚠️ **GitHub Actions**: Using wrong deployment file (ip.yml instead of production.yml)

## 🚨 Deployment Issue Fixed
**Problem**: GitHub Actions workflow was trying to use `docker-compose.ip.yml` (IP-based deployment) instead of `docker-compose.production.yml` (domain-based deployment) for the production domain `api.climateriskplan.com`.

**Symptoms**: 
- Health check timeouts after 20 attempts
- Containers failing to start properly
- Missing docker-compose.ip.yml file errors

**Solution**: Use the correct deployment script for production domain.

## Quick Fix (Run on server)
```bash
# Use the production fix script
./fix-production.sh

# Or manually force production deployment
docker compose -f docker-compose.production.yml down
docker compose -f docker-compose.production.yml up -d --build
```

## Files

- `Caddyfile.production` - Production Caddy configuration for api.climateriskplan.com
- `docker-compose.production.yml` - Production Docker Compose configuration  
- `deploy.sh` - Main deployment script
- `diagnose.sh` - HTTPS troubleshooting script

## Quick Start

1. **Deploy to production:**
   ```bash
   ./deploy.sh
   ```

2. **Diagnose HTTPS issues:**
   ```bash
   ./diagnose.sh
   ```

3. **Fix HTTPS certificate:**
   ```bash
   # Restart Caddy to regenerate certificates
   docker compose -f docker-compose.production.yml restart caddy
   
   # Or full restart if needed
   docker compose -f docker-compose.production.yml down
   docker compose -f docker-compose.production.yml up -d
   ```

4. **View logs:**
   ```bash
   docker compose -f docker-compose.production.yml logs -f caddy
   ```

## Requirements

- Domain `api.climateriskplan.com` must point to your server's IP
- Ports 80 and 443 must be open
- Docker and Docker Compose installed

## Troubleshooting

If HTTPS isn't working:

1. Run `./diagnose.sh` to identify issues
2. Check DNS: `dig api.climateriskplan.com`
3. Check containers: `docker ps`
4. Check logs: `docker logs photogrammetry-caddy`

Common fixes:
- Update DNS A record
- Restart containers: `./deploy.sh`
- Wait 5-10 minutes for certificate issuance
