#!/bin/bash
set -euo pipefail

echo "Upgrading pip..."
pip install --upgrade pip || {
    echo "Failed to upgrade pip"
    exit 1
}

if [ -f requirements.txt ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt || {
        echo "Failed to install requirements"
        exit 1
    }
else
    echo "No requirements.txt found, skipping package installation"
fi

echo "Setting up terminal prompt..."
cat << 'EOF' >> ~/.bashrc
# Function to get git branch
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

# Color definitions
BLUE='\[\033[34m\]'
GREEN='\[\033[32m\]'
YELLOW='\[\033[33m\]'
RESET='\[\033[00m\]'

# Set prompt with current directory and git branch
export PS1="${BLUE}\W${RESET}${YELLOW}\$(parse_git_branch)${RESET}${GREEN} $ ${RESET}"
EOF

echo "Setup completed successfully!" 