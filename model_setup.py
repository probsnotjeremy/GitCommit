# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python (vLLM)
#     language: python
#     name: vllm-env
# ---

# +
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from dataset_prep import ds
import os


# --------------------------
# Global vars
# --------------------------
history = []
filename = "TestResults.json"

# --------------------------
# Setup model + tokenizer
# --------------------------
model_id = "microsoft/phi-2"

adapter_path = "./qlora_phi2/qlora_phi2_best"


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False
)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16",
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16,
    quantization_config=bnb_config
)

# Check for adapter
if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
else:
    print("No adapter found, using base model only.")
    model = base_model

# --------------------------
# LLM wrapper
# --------------------------
def ask_phi2(user_input, max_new_tokens=128):
    global history
    history.append(f"User: {user_input}")
    prompt = "\n".join(history) + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only new tokens (not the prompt)
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print(f"Generated {len(gen_ids)} tokens")

    history.append(f"Assistant: {response}")
    return response


def send_to_llm(question):
    global history
    history = []  # reset history per question
    return ask_phi2(question)

# --------------------------
# Tokenizer
# --------------------------

raw_dataset = ds

def tokenize_function(examples):
    inputs = [f"Question: {q}" for q in examples["query"]]
    targets = [a for a in examples["answer"]]

    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    labels = tokenizer(
        targets,
        padding="max_length",
        truncation=True,
        max_length=128
    )["input_ids"]

    # Mask out padding
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
    model_inputs["labels"] = labels

    return model_inputs



tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["query", "answer"]
)


# --------------------------
# LoRA setup
# --------------------------

def make_lora_config(r=16, alpha=32, dropout=0.05):
    print("QLoRA-ready model initialized.")
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

lora_model = get_peft_model(model, make_lora_config())

# !jupytext --set-formats ipynb,py:light --sync model_setup.ipynb

# -


