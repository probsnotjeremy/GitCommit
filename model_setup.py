#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model

# --------------------------
# Global vars
# --------------------------
history = []
filename = "TestResults.json"

# --------------------------
# Setup model + tokenizer
# --------------------------
model_id = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16,
    quantization_config=bnb_config
)

# --------------------------
# LLM wrapper
# --------------------------
def ask_phi2(user_input, max_new_tokens=64):
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
    return ask_phi2(question, max_new_tokens=64)

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

model = get_peft_model(model, make_lora_config())
