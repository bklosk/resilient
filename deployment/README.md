# Production Deployment

Clean deployment configuration for the Photogrammetry API at `api.climateriskplan.com`.

## Status
✅ **Production**: Working at https://api.climateriskplan.com  
✅ **HTTP**: Working at http://api.climateriskplan.com  
✅ **SSL**: Let's Encrypt certificates active  
✅ **Health**: API responding normally  

## Files

- `docker-compose.production.yml` - Production deployment configuration
- `Caddyfile.production` - Caddy reverse proxy with automatic HTTPS
- `deploy.sh` - Main deployment script
- `diagnose.sh` - Diagnostic and troubleshooting tool

## Usage

### Deploy
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

1. **Check containers**: `docker compose -f docker-compose.production.yml ps`
2. **Check logs**: `docker compose -f docker-compose.production.yml logs caddy`
3. **Restart for certificate issues**: `docker compose -f docker-compose.production.yml restart caddy`
4. **Full restart**: `./deploy.sh`

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
