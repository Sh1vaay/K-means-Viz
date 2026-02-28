#!/bin/bash
# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -r requirements.txt
fi

echo "Starting K-Meanr..."
echo "Go to http://localhost:5000 in your browser."
./venv/bin/python app.py
