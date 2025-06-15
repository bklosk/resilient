# Production Deployment

Minimal production deployment for `api.climateriskplan.com`.

## Quick Commands

```bash
./deploy.sh              # Deploy
docker compose ps        # Status  
docker compose logs -f   # Logs
docker compose restart   # Restart
docker compose down      # Stop
```

## Files

- `docker-compose.yml` - Production configuration
- `Caddyfile.production` - HTTPS proxy configuration  
- `deploy.sh` - Deployment script
- `diagnose.sh` - Diagnostic tool

## Architecture

- **FastAPI** app on port 8000 (internal)
- **Caddy** proxy on ports 80/443 (public)
- **Automatic HTTPS** via Let's Encrypt
- **Health monitoring** with 60s startup window

## Requirements

- Domain pointing to server IP
- Ports 80/443 open
- Docker Compose v2+

## Troubleshooting

- **Certificates**: `docker compose restart caddy`
- **Clear cache**: `docker volume rm $(docker volume ls -q | grep caddy)`
- **Full reset**: `docker compose down && ./deploy.sh`
