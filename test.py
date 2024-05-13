import os
from dotenv import load_dotenv

load_dotenv()


# Check
if HF_READ_PASSKEY is None or HF_WRITE_PASSKEY is None:
    raise ValueError("One or more environment variables are missing.")

HF_READ_PASSKEY = os.getenv('HF_READ_PASSKEY')
HF_WRITE_PASSKEY = os.getenv('HF_WRITE_PASSKEY')

# Print to verify the variables are loaded (for debugging purposes)
print("Huggingface Read Passkey:", HF_READ_PASSKEY)
print("Huggingface Write Passkey:", HF_READ_PASSKEY)
