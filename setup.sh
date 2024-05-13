#!/bin/bash
# Stop execution on any error
set -e

# Define user-specific variables
GITHUB_USERNAME="efnogron"
GITHUB_EMAIL="42118609+efnogron@users.noreply.github.com"

# Move to workspace for permanent storage
echo "Moving to workspace directory..."
cd /workspace/

# Configure Git for SSH operations
echo "Configuring Git..."
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"

# Generate SSH Key
echo "Generating SSH key..."
ssh-keygen -t rsa -b 4096 -C "$GITHUB_EMAIL" -f ~/.ssh/id_rsa -N ""

# Start the SSH agent in the background
echo "Starting SSH agent..."
eval "$(ssh-agent -s)"

# Add SSH key to the SSH agent
echo "Adding SSH key to the agent..."
ssh-add ~/.ssh/id_rsa

# Display the SSH key and prompt for manual action
echo "Copy the following SSH key to GitHub under Settings > SSH and GPG keys:"
cat ~/.ssh/id_rsa.pub
echo "Press any key to continue after adding SSH key to GitHub..."
read -n 1 -s

# Clone the repository
echo "Cloning repository..."
git clone https://github.com/$GITHUB_USERNAME/tiro_finetune.git

# Move to the project directory
echo "Changing to project directory..."
cd tiro_finetune

# Set up Python environment
echo "Setting up Python environment..."
python -m venv tiroEnv
source tiroEnv/bin/activate

# Upgrade pip and install requirements
echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Login to Hugging Face
echo "Logging into Hugging Face..."
huggingface-cli login

echo "Setup completed successfully."
