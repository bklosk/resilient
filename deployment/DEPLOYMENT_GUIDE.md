# Quick Deployment Guide

## Automatic Deployment (GitHub Actions)

The application automatically deploys to your DigitalOcean server when you push to the `main` branch.

**What happens:**
1. Code is packaged and uploaded to the server
2. Nginx is installed and configured as a reverse proxy
3. Python dependencies are installed
4. FastAPI application starts on localhost:8000
5. Nginx serves requests on port 80 and proxies to the app

## Manual HTTPS Setup

After the initial deployment, you can add HTTPS support:

```bash
# SSH into your server
ssh user@your-server-ip

# Navigate to the application directory
cd /opt/photogrammetry

# Run the HTTPS setup script
sudo ./deployment/scripts/setup-https.sh your-domain.com your-email@domain.com
```

## Health Monitoring

Check if everything is working:

```bash
# On the server
./deployment/scripts/check-deployment.sh
```

## Troubleshooting

### Application not responding
```bash
# Check if the app is running
pgrep -f uvicorn

# Restart the application
sudo systemctl restart nginx
cd /opt/photogrammetry
killall uvicorn
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 &
```

### Nginx issues
```bash
# Check nginx status
sudo systemctl status nginx

# Test nginx configuration
sudo nginx -t

# Check nginx logs
sudo tail -f /var/log/nginx/error.log
```

### Port issues
```bash
# Check what's listening on ports
sudo netstat -tlnp | grep -E ':(80|443|8000)'
```
