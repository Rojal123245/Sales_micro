#!/bin/bash

# Check if the API is already running
if ! curl -s http://localhost:8084/health > /dev/null; then
    echo "Starting API service..."
    # Start the API in the background
    python api_gateway/app.py --port 8084 &
    API_PID=$!

    # Wait for API to start
    echo "Waiting for API to start..."
    while ! curl -s http://localhost:8084/health > /dev/null; do
        sleep 1
    done
    echo "API service started successfully!"
else
    echo "API service is already running."
fi

# Start Streamlit
echo "Starting Streamlit application..."
streamlit run streamlit_app.py

# If we started the API, kill it when Streamlit exits
if [ -n "$API_PID" ]; then
    echo "Shutting down API service..."
    kill $API_PID
fi
