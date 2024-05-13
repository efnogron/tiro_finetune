# tiro_finetune

# move to workspace for permanent storage
cd /workspace/

## How to setup SSH to Runpod
git config --global user.name "efnogron"
git config --global user.email "42118609+efnogron@users.noreply.github.com"

# Generate SSH Key
ssh-keygen -t rsa -b 4096 -C "42118609+efnogron@users.noreply.github.com"

# Add SSH Key to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# Display SSH Key
cat ~/.ssh/id_rsa.pub
copy results into git SSH section


git clone https://github.com/efnogron/tiro_finetune.git

#move to tiro directory
cd tiro_finetune

# Install requirements
python -m venv tiroEnv
source tiroEnv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Log Into huggingface
huggingface-cli login
