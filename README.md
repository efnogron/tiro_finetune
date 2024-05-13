# tiro_finetune

## How to setup SSH to Runpod
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Generate SSH Key
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"

# Add SSH Key to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# Display SSH Key
cat ~/.ssh/id_rsa.pub
copy results into git SSH section

git clone YOUR_REPOSITORY_URL

# Install requirements
python -m venv tiroEnv
source tiroEnv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Log Into huggingface
huggingface-cli login
