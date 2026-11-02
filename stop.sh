#!/usr/bin/env bash
# Stop the server ./start.sh started. Weights stay in MODELS_DIR.
docker rm -f nanbeige-maxq >/dev/null 2>&1 && echo "stopped nanbeige-maxq" || echo "nanbeige-maxq was not running"
