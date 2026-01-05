#!/bin/bash
# Start base image services (Jupyter/SSH) in background
/start.sh &

# Wait for services to start
sleep 2

# Run your application
python /app/main.py

# Wait for background processes
wait