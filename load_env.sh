#!/bin/bash
# Helper script to load .env file from project root

# Get project root directory
if [ -f ".env" ]; then
    # If .env exists in current directory
    PROJECT_ROOT="$(pwd)"
elif [ -f "../../../.env" ]; then
    # If called from scripts/* subdirectory
    PROJECT_ROOT="$(cd ../../.. && pwd)"
elif [ -f "../../.env" ]; then
    # If called from scripts subdirectory  
    PROJECT_ROOT="$(cd ../.. && pwd)"
else
    # Try to find .env by going up directories
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd 2>/dev/null || cd "$SCRIPT_DIR/../../.." && pwd 2>/dev/null || pwd)"
fi

# Try .env first, then config.env
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE="$PROJECT_ROOT/config.env"
fi

if [ -f "$ENV_FILE" ]; then
    # Load .env/config.env file and export variables
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: config file not found at $PROJECT_ROOT/.env or $PROJECT_ROOT/config.env" >&2
fi

