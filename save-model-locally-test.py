# this is the tiro-tinetune-8b.py script, but it saves the model while finetuning to avoid duplicate downloads.

import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch

load_dotenv()

HF_READ_PASSKEY = os.getenv('HF_READ_PASSKEY')
HF_WRITE_PASSKEY = os.getenv('HF_WRITE_PASSKEY')

# Check if environment variables are set correctly
if HF_READ_PASSKEY is None or HF_WRITE_PASSKEY is None:
    raise ValueError("One or more environment variables are missing.")

# Print to verify the variables are loaded (for debugging purposes)
print("Huggingface Read Passkey:", HF_READ_PASSKEY)
print("Huggingface Write Passkey:", HF_READ_PASSKEY)

max_seq_length = 4096  # automatically does RoPE Scaling internally, can choose any value
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.
load_in_8bit = False

model_name = "meta-llama/Meta-Llama-3-70B"
local_model_path = "local_model"

# Check if the model is already downloaded
if os.path.exists(local_model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        local_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit
    )
    print("Model loaded from local path.")
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit
    )
    model.save_pretrained(local_model_path)  # Save the model locally
    tokenizer.save_pretrained(local_model_path)
    print("Model downloaded and saved locally.")

print("Model and tokenizer successfully loaded.")
print("Model architecture:", model)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tiro_prompt = """
### Instruction:
{instruction}

### Here is the original:
{question}

### Answer:
{answer}

### Extra Information:
{extra}

### User Input:
{input}

### Expected Response:
{output}
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    questions = examples["question"]
    answers = examples["answer"]
    extras = examples["extra"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, question, answer, extra, input, output in zip(instructions, questions, answers, extras, inputs, outputs):
        # Fill the template with data from the dataset
        text = tiro_prompt.format(
            instruction=instruction,
            question=question,
            answer=answer,
            extra=extra,
            input=input,
            output=output
        ) + EOS_TOKEN  # Add EOS token to mark the end of the text
        texts.append(text)

    return {"text": texts}
pass

from datasets import load_dataset
dataset = load_dataset("OG-Tiro/Finetune_Evaluate_Answer", token=HF_READ_PASSKEY, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 2, # will override epochs if max steps is given
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("Llama3dumbbabytiro") # Local saving
tokenizer.save_pretrained("Llama3dumbbabytiro")
model.push_to_hub("OG-Tiro/Llama3dumbbabytiro", token = HF_WRITE_PASSKEY) # Online saving
tokenizer.push_to_hub("OG-Tiro/Llama3dumbbabytiro", token = HF_WRITE_PASSKEY) # Online saving

# Merge to 16bit
if False: model.save_pretrained_merged("Llama3dumbbabytiro", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("OG-Tiro/Llama3dumbbabytiro", tokenizer, save_method = "merged_16bit", token = HF_WRITE_PASSKEY) #highest available quality

# Merge to 4bit
if False: model.save_pretrained_merged("Llama3dumbbabytiro", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("OG-Tiro/Llama3dumbbabytiro", tokenizer, save_method = "merged_4bit", token = HF_WRITE_PASSKEY)

# Just LoRA adapters
if False: model.save_pretrained_merged("Llama3dumbbabytiro", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("OG-Tiro/Llama3dumbbabytiro", tokenizer, save_method = "lora", token = HF_WRITE_PASSKEY)

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("Llama3dumbbabytiro", tokenizer,)
if False: model.push_to_hub_gguf("OG-Tiro/Llama3dumbbabytiro", tokenizer, token = HF_WRITE_PASSKEY)

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("Llama3dumbbabytiro", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("OG-Tiro/Llama3dumbbabytiro", tokenizer, quantization_method = "f16", token = HF_WRITE_PASSKEY)

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("Llama3dumbbabytiro", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("OG-Tiro/Llama3dumbbabytiro", tokenizer, quantization_method = "q4_k_m", token = HF_WRITE_PASSKEY)
