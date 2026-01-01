import argparse
import json
import time
import hashlib
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hash_messages(messages):
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (e.g., 03_runs/lora_s1)")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ✅ 4bit 로딩(노트북 VRAM에서 가장 안정적)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # ✅ LoRA 어댑터 적용(옵션)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)

    gen_params = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature > 0),
    )

    rows = []
    for ex in load_jsonl(Path(args.eval)):
        messages = ex["messages"]

        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, **gen_params)
        text = tok.decode(output[0], skip_special_tokens=True)

        resp = text[len(prompt):].lstrip() if text.startswith(prompt) else text

        rows.append({
            "id": ex.get("id"),
            "category": ex.get("category"),
            "prompt_hash": hash_messages(messages),
            "messages": messages,
            "model_name": args.model,
            "adapter": args.adapter,
            "gen_params": gen_params,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "response": resp,
        })

    save_jsonl(Path(args.out), rows)
    print(f"Saved: {args.out} (n={len(rows)})")


if __name__ == "__main__":
    main()
