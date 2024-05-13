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
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC1goOw13Frkd8aj4gBF7OuKX4tMLKMKFnQo3cmcoTTu46wBTiV36WzCCu1moKX4BnIB8Bp/i7T/JKbvnxco0E3DfE7lWf9ZnLpNbpl7yhJ8Iir+aLlHQjOiEsgqHEfVRSPOJPJer6RzKpwkTpShOUXtHhpekBxkpmrULgx3KSD3ssNOd+k69JbFKe8bnSJTjK7TjBCLfy2VlhMFnwibgR4CHRFuBzTA2NVg/7JDXrQagkl7/NllOfEVcVV+CEjevrXqBKPzkhWJDvUhJsHPsmSflKNOGVSJN0KmgK1QtumAvd/dGH6znBshyL8KGrK4FUL/xZ3kh5bMLKPo+RKElRz+rX3p8l3wzrQnh3rExQC/iKiQgPrGiXLVM5PtcE0G/y7WBuHlcgW+449iHoPNchzXkZWd6hU4X2dwiHDpl3xgRIccnVDvmBkH7oDXNi0isLtrQsQ9PaTdxpsKrL9g6JkkBLtmN+MAep3ci6x+VOG2DAiilv3abKNu/D0ny/WFxdS1JHGUKQpaNwEcB/mDvoOwb/hSe1T9hhEoZeuGXNumRiO+AKfpzikICeAGHg/pj2l2n4mWvDwwJuN7fB7/1Qvlu5koPYSzJnXJgfcPn2W7zLh15wsUvLqeDX19a0Zd9eRLTHg+RikNO7uCydQpuh1v1ebGPBWkzq5YNJ3H6et0w== 42118609+efnogron@users.noreply.github.com

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
