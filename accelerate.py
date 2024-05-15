import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Initialize the accelerator
accelerator = Accelerator()

load_dotenv()

HF_READ_PASSKEY = os.getenv('HF_READ_PASSKEY')
HF_WRITE_PASSKEY = os.getenv('HF_WRITE_PASSKEY')

# Check if environment variables are set correctly
if HF_READ_PASSKEY is None or HF_WRITE_PASSKEY is None:
    raise ValueError("One or more environment variables are missing.")

max_seq_length = 4096  # automatically does RoPE Scaling internally, can choose any value
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype
)

print("Model and tokenizer successfully loaded.")
print("Model architecture:", model)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
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

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for instruction, question, answer, extra, input, output in zip(examples["instruction"], examples["question"], examples["answer"], examples["extra"], examples["input"], examples["output"]):
        text = tiro_prompt.format(
            instruction=instruction,
            question=question,
            answer=answer,
            extra=extra,
            input=input,
            output=output
        ) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("OG-Tiro/Finetune_Evaluate_Answer", token=HF_READ_PASSKEY, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=2,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args
)


# Prepare model, optimizer, and dataloader with accelerator
trainer.model, trainer.optimizer, trainer.train_dataloader = accelerator.prepare(
    trainer.model, trainer.optimizer, trainer.train_dataloader
)

# Training loop
trainer.model.train()
for epoch in range(1):
    for step, batch in enumerate(trainer.train_dataloader):
        trainer.optimizer.zero_grad()
        inputs = {k: v.to(accelerator.device) for k, v in batch.items() if k in tokenizer.model_input_names}
        outputs = trainer.model(**inputs)
        loss = outputs.loss
        accelerator.backward(loss)
        trainer.optimizer.step()
        if step % training_args.logging_steps == 0:
            print(f"Step {step}, Loss: {loss.item()}")

# Save the model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(trainer.model)
unwrapped_model.save_pretrained("Llama3dumbbabytiro")
tokenizer.save_pretrained("Llama3dumbbabytiro")
unwrapped_model.push_to_hub("OG-Tiro/Llama3dumbbabytiro", token=HF_WRITE_PASSKEY)
tokenizer.push_to_hub("OG-Tiro/Llama3dumbbabytiro", token=HF_WRITE_PASSKEY)

# Memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")