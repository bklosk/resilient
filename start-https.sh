#!/bin/bash
./setup-ssl.sh $1
docker-compose up -d
echo "HTTPS server starting at https://localhost (or https://$1 if domain specified)"
