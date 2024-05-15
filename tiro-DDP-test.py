#cannot run this via python, must run via accelarate: !accelerate launch tiro-finetune-70b.py

import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from accelerate import PartialState

load_dotenv()

HF_READ_PASSKEY = os.getenv('HF_READ_PASSKEY')
HF_WRITE_PASSKEY = os.getenv('HF_WRITE_PASSKEY')

# Check if environment variables are set correctly
if HF_READ_PASSKEY is None or HF_WRITE_PASSKEY is None:
    raise ValueError("One or more environment variables are missing.")

# Print to verify the variables are loaded (for debugging purposes)
print("Huggingface Read Passkey:", HF_READ_PASSKEY)
print("Huggingface Write Passkey:", HF_READ_PASSKEY)

# Define Prompt for training data
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

# Load Dataset from Huggingface
dataset = load_dataset("OG-Tiro/Finetune_Evaluate_Answer_Cleaned", token=HF_READ_PASSKEY, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

#model and tokenizer arguments
max_seq_length = 4096 # automatically does RoPE Scaling internally, can choose any value
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
load_in_8bit = False
device_map = "DDP" # Sets up distributed data parallel using accelerate

if device_map == "DDP":
    device_string = PartialState().process__index
    device_map={'':device_string}

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit", # useless
    "unsloth/llama-3-70b-Instruct-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit", #
] # More models at https://huggingface.co/unsloth


model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "unsloth/llama-3-8b-bnb-4bit",
    #model_name = "unsloth/llama-3-8b", # 8 Bit version
    model_name = "unsloth/llama-3-70b-Instruct-bnb-4bit",
    #model_name = "meta-llama/Meta-Llama-3-70B",
    #model_name = "unsloth/llama-3-70b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit, # i guess this must be changed to load in 8 bit?
    load_in_8bit = load_in_8bit
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

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
    device_map = device_map,
)





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
        #max_steps = 200, # will override epochs if max steps is given
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
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

model.save_pretrained("bigbabytiro70b") # Local saving
tokenizer.save_pretrained("bigbabytiro70b")
model.push_to_hub("OG-Tiro/bigbabytiro70b", token = HF_WRITE_PASSKEY) # Online saving
tokenizer.push_to_hub("OG-Tiro/bigbabytiro70b", token = HF_WRITE_PASSKEY) # Online saving

# Merge the LoRA adapters with the uncompressed model
merged_model = FastLanguageModel.merge_lora_adapters(
    model=model,
    lora_adapters_path="outputs"
)

# Save and upload the merged model
merged_model.save_pretrained("bigbabytiro70b_merged")
tokenizer.save_pretrained("bigbabytiro70b_merged")
merged_model.push_to_hub("OG-Tiro/bigbabytiro70b_merged", token=HF_WRITE_PASSKEY)
tokenizer.push_to_hub("OG-Tiro/bigbabytiro70b_merged", token=HF_WRITE_PASSKEY)

# Merge to 16bit
if False: model.save_pretrained_merged("bigbabytiro70b", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("OG-Tiro/bigbabytiro70b", tokenizer, save_method = "merged_16bit", token = HF_WRITE_PASSKEY) #highest available quality

# Merge to 4bit
if False: model.save_pretrained_merged("bigbabytiro70b", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("OG-Tiro/bigbabytiro70b", tokenizer, save_method = "merged_4bit", token = HF_WRITE_PASSKEY)

# Just LoRA adapters
if False: model.save_pretrained_merged("bigbabytiro70b", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("OG-Tiro/bigbabytiro70b", tokenizer, save_method = "lora", token = HF_WRITE_PASSKEY)

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("bigbabytiro70b", tokenizer,)# quantization method missing!!!
if False: model.push_to_hub_gguf("OG-Tiro/bigbabytiro70b", tokenizer, token = HF_WRITE_PASSKEY)

# Save to 16bit GGUF
if True: model.save_pretrained_gguf("bigbabytiro70b", tokenizer, quantization_method = "f16")
if True: model.push_to_hub_gguf("OG-Tiro/bigbabytiro70b", tokenizer, quantization_method = "f16", token = HF_WRITE_PASSKEY)

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("bigbabytiro70b", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("OG-Tiro/bigbabytiro70b", tokenizer, quantization_method = "q4_k_m", token = HF_WRITE_PASSKEY)