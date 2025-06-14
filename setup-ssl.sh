#!/bin/bash
DOMAIN=${1:-localhost}
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -keyout ssl/privkey.pem -out ssl/fullchain.pem -sha256 -days 365 -nodes -subj "/CN=$DOMAIN"
echo "SSL certificates generated for $DOMAIN"