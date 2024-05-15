.
├── Dockerfile
├── README.md
├── gpu-config.py
├── jupyter_notebooks
│   └── Tiro_+_Llama_3_8b_Finetune_070524.ipynb
├── requirements.txt
├── setup.sh
├── test.py
├── tiro-finetune-70b.py
└── tiro-finetune-8b.py

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

# Initialize Accelerate
accelerate config

# Monitor GPU usage
watch -n 2 nvidia-smi
