# Deployment Configuration

This directory contains all deployment-related configuration files and scripts for the Photogrammetry API.

## Directory Structure

```
deployment/
├── nginx/           # Nginx configuration files
│   ├── http.conf    # Basic HTTP configuration (used by GitHub Actions)
│   └── https.conf   # HTTPS configuration template with Let's Encrypt
├── scripts/         # Deployment and management scripts
│   ├── setup-https.sh       # Automated HTTPS setup with Let's Encrypt
│   └── check-deployment.sh  # Deployment status monitoring script
└── README.md        # This file
```

## Nginx Configurations

### `nginx/http.conf`
- Basic HTTP-only configuration
- Used by the GitHub Actions deployment workflow
- Listens on port 80 and proxies to localhost:8000
- Optimized for large file uploads (100MB limit)
- Includes gzip compression and performance optimizations

### `nginx/https.conf`
- Complete HTTPS configuration template
- Includes security headers and SSL best practices
- Ready for Let's Encrypt certificates
- Redirects HTTP to HTTPS when enabled

## Scripts

### `scripts/setup-https.sh`
**Usage:** `sudo ./setup-https.sh <domain> [email]`

Automated script that:
- Installs certbot if not present
- Obtains Let's Encrypt SSL certificates
- Configures nginx for HTTPS
- Sets up automatic certificate renewal
- Enables HTTP to HTTPS redirects

**Example:**
```bash
sudo ./scripts/setup-https.sh api.example.com admin@example.com
```

### `scripts/check-deployment.sh`
**Usage:** `./check-deployment.sh`

Status monitoring script that checks:
- Nginx service status and configuration validity
- FastAPI application status
- Health endpoint accessibility (local and via proxy)
- HTTPS configuration (if certificates exist)
- Active listening ports
- Recent nginx logs

## Deployment Flow

1. **Initial Deployment:** GitHub Actions uses `nginx/http.conf` configuration
2. **HTTPS Setup:** Run `setup-https.sh` with your domain when ready
3. **Monitoring:** Use `check-deployment.sh` to verify deployment health

## Security Notes

- The FastAPI application binds only to `127.0.0.1:8000` (localhost)
- All external access goes through the nginx reverse proxy
- HTTPS configuration includes modern security headers
- Large file uploads are supported for point cloud data processing
