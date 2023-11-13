from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          default_data_collator, get_linear_schedule_with_warmup)
from peft import (get_peft_config, get_peft_model,
                  get_peft_model_state_dict, PrefixTuningConfig, TaskType)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os


def train(train_dataset, eval_dataset, model_path, max_length=128, lr=1e-2, num_epochs=5, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 128
    train_dataloader = DataLoader(train_dataset, shuffle=True, 
                                  collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, 
                                 batch_size=batch_size, pin_memory=True)

    peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = get_peft_model(model, peft_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        return None