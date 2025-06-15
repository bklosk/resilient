# Deployment Guide

This directory contains the simplified deployment configuration for the Photogrammetry API.

## Files

- `Caddyfile.production` - Production Caddy configuration for api.climateriskplan.com
- `docker-compose.production.yml` - Production Docker Compose configuration
- `deploy.sh` - Main deployment script
- `diagnose.sh` - HTTPS troubleshooting script
- `backup/` - Backup of old deployment files

## Quick Start

1. **Deploy to production:**
   ```bash
   ./deploy.sh
   ```

2. **Diagnose HTTPS issues:**
   ```bash
   ./diagnose.sh
   ```

3. **View logs:**
   ```bash
   docker compose -f docker-compose.production.yml logs -f
   ```

4. **Stop deployment:**
   ```bash
   docker compose -f docker-compose.production.yml down
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
