# Use the specific RunPod PyTorch image
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Start with a base image that includes Python and the necessary tools
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install git, SSH, and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    python3-venv

# Configure git
RUN git config --global user.name "efnogron" \
 && git config --global user.email "42118609+efnogron@users.noreply.github.com"

# Setup SSH Key (Here you might want to use a more secure method to handle the keys)
RUN ssh-keygen -t rsa -b 4096 -C "42118609+efnogron@users.noreply.github.com" -N "" -f ~/.ssh/id_rsa

# Add SSH key to the SSH agent
RUN eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_rsa

# Clone your repository
RUN git clone https://github.com/efnogron/tiro_finetune.git

# Set the working directory to your project folder
WORKDIR /workspace/tiro_finetune

# Setup a virtual environment and install dependencies
RUN python3 -m venv tiroEnv \
 && . tiroEnv/bin/activate \
 && python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# Command to run on container start
CMD ["bash"]

# Set the working directory in the Docker container
WORKDIR /workspace/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .