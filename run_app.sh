#!/bin/bash

# Kill any processes using port 8084
echo "Killing any processes using port 8084..."
lsof -ti:8084 | xargs kill -9 2>/dev/null || true

# Start the API
echo "Starting API on port 8084..."
python api_gateway/app.py --port 8084 &
API_PID=$!

# Wait for API to start
echo "Waiting for API to start..."
sleep 5

# Start Streamlit
echo "Starting Streamlit application..."
streamlit run streamlit_app.py

# If we started the API, kill it when Streamlit exits
if [ -n "$API_PID" ]; then
    echo "Shutting down API service..."
    kill $API_PID
fi
