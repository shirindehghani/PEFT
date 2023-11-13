import os, torch, logging
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          BitsAndBytesConfig, HfArgumentParser, 
                          TrainingArguments, pipeline)
from peft import (LoraConfig, PeftModel)
from trl import SFTTrainer
import pandas as pd

def train(model_name, dataset_path, is_data_ready=True):
    if is_data_ready==True:
        training_data=load_dataset(dataset_path, split="train")
    else:
        training_data=pd.read_csv(dataset_path)

    llama_tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                    trust_remote_code=True)
    llama_tokenizer.pad_token=llama_tokenizer.eos_token
    llama_tokenizer.padding_side="right"
    quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False)
    base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quant_config,
                    device_map={"": 0})
    base_model.config.use_chache=False
    base_model.config.pretraining_tp=1

    peft_parameters = LoraConfig(
                    lora_alpha=16,
                    lora_dropout=0.1,
                    r=8,
                    bias="none",
                    task_type="CAUSAL_LM")
    train_params = TrainingArguments(
        output_dir="./results_modified", num_train_epochs=1,
        per_device_train_batch_size=4, gradient_accumulation_steps=1,
        optim="paged_adamw_32bit", save_steps=25,
        logging_steps=25, learning_rate=2e-4, weight_decay=0.001,
        fp16=False, bf16=False, max_grad_norm=0.3,
        max_steps=-1, warmup_ratio=0.03, group_by_length=True, lr_scheduler_type="constant"
    )
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params)
    fine_tuning.train()
    return fine_tuning