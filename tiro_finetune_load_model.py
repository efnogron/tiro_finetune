from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # automatically does RoPE Scaling internally, can choose any value
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
load_in_8bit = True

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit", # useless
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit", #
] # More models at https://huggingface.co/unsloth


model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "unsloth/llama-3-8b-bnb-4bit",
    model_name = "unsloth/llama-3-8b", # 8 Bit version
    #model_name = "unsloth/llama-3-70b-bnb-4bit",
    #model_name = "unsloth/llama-3-70b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    #load_in_4bit = load_in_4bit, # i guess this must be changed to load in 8 bit?
    #load_in_8bit = load_in_8bit
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
dataset = load_dataset("OG-Tiro/Finetune_Evaluate_Answer", token="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", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
