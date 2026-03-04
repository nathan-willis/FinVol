#!/bin/bash

# Check if at least one argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <message>"
    exit 1
fi

# Define the directory name
DIRECTORY_NAME="$1"

# Create the directory
mkdir "$DIRECTORY_NAME"
mkdir "$DIRECTORY_NAME/sims"
mkdir "$DIRECTORY_NAME/solutions"
mkdir "$DIRECTORY_NAME/solutions/plots"
mkdir "$DIRECTORY_NAME/solutions/postData"
mkdir "$DIRECTORY_NAME/solutions/videos"

echo "Directory '$DIRECTORY_NAME' created."

