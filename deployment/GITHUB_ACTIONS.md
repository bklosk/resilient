# GitHub Actions Integration

## Quick Fix for Deployment Issues

Replace the deployment section in `.github/workflows/deploy-dev.yml` with:

```yaml
# Replace the complex deployment logic with:
cd /opt/photogrammetry/deployment
./auto-deploy.sh
```

## Current Auto-Deploy Features

The `auto-deploy.sh` script handles:

1. **Production First**: Tries `docker-compose.production.yml`
   - Domain-based HTTPS with Let's Encrypt
   - Best option for api.climateriskplan.com

2. **IP Fallback**: Uses `docker-compose.ip.yml` 
   - Self-signed certificates
   - Works when domain certificates fail

3. **Development Fallback**: Uses `docker-compose.dev.yml`
   - HTTP-only on port 8080
   - Used when ports 80/443 are blocked

4. **Automatic Testing**: Tests each deployment
5. **Clean Transitions**: Stops previous deployments properly
6. **Clear Reporting**: Shows exactly what's working

## Benefits

- ✅ **No more manual emergency fixes needed**
- ✅ **Handles certificate issues automatically**
- ✅ **Falls back gracefully when ports are in use**
- ✅ **Works with existing GitHub Actions secrets**
- ✅ **Provides clear status for CI/CD**

## Migration

The GitHub Actions workflow can be simplified to:

```yaml
- name: Deploy Application
  run: |
    cd /opt/photogrammetry/deployment
    ./auto-deploy.sh
```

This replaces all the complex port checking, container cleanup, and deployment strategy logic.
