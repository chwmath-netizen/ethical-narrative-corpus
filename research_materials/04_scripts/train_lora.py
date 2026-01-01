import argparse, json
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def read_text_jsonl(path):
    texts=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            obj=json.loads(line)
            texts.append(obj["text"])
    return texts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=60)
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model=AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_4bit=True,  # QLoRA (bitsandbytes 필요)
    )
    model = prepare_model_for_kbit_training(model)
    
    lora=LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"]
    )
    model=get_peft_model(model, lora)
    
    texts=read_text_jsonl(args.train_jsonl)
    ds=Dataset.from_dict({"text": texts})
    
    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)
    
    ds=ds.map(tok_fn, batched=True, remove_columns=["text"])
    
    collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    
    training_args=TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args.steps,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=args.steps,
        fp16=True,
        report_to="none",
    )
    
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )
    
    trainer.train()
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("saved adapter to:", args.out_dir)

if __name__=="__main__":
    main()