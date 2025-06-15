# Production Deployment

Clean deployment configuration for the Photogrammetry API at `api.climateriskplan.com`.

## Status
⚠️ **HTTPS**: May need recovery after cleanup  
✅ **HTTP**: Should be working  
⚠️ **SSL**: May need certificate regeneration  

## Emergency Recovery
If HTTPS stopped working after cleanup:
```bash
./emergency-fix.sh
```

## Files

- `docker-compose.production.yml` - Production deployment configuration
- `docker-compose.ip.yml` - IP-based fallback deployment
- `Caddyfile.production` - Production Caddy with automatic HTTPS
- `Caddyfile.ip` - IP-based Caddy with self-signed certificates
- `deploy.sh` - Main deployment script
- `diagnose.sh` - Diagnostic and troubleshooting tool
- `emergency-fix.sh` - Emergency recovery script

## Usage

### Emergency Fix (if HTTPS broken)
```bash
./emergency-fix.sh
```

### Normal Deploy
```bash
./deploy.sh
```

### Check Status
```bash
./diagnose.sh
```

### View Logs
```bash
docker compose -f docker-compose.production.yml logs -f
```

### Restart Services
```bash
docker compose -f docker-compose.production.yml restart
```

### Stop Deployment
```bash
docker compose -f docker-compose.production.yml down
```

## Requirements

- Domain `api.climateriskplan.com` pointing to server IP
- Ports 80 and 443 open
- Docker and Docker Compose installed

## Troubleshooting

1. **HTTPS broken**: Run `./emergency-fix.sh`
2. **Check containers**: `docker compose -f docker-compose.production.yml ps`
3. **Check logs**: `docker compose -f docker-compose.production.yml logs caddy`
4. **Restart for certificate issues**: `docker compose -f docker-compose.production.yml restart caddy`
5. **Full restart**: `./deploy.sh`

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
