"""Stage C SFT training for targeted and comparison adapter arms."""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from stage_c_adapters import (
    count_trainable_parameters,
    save_head_sliced_adapter,
    wrap_head_sliced_adapters,
)
from stage_c_eval import build_chat_prompt, load_jsonl
from stage_c_heads import random_heads_outside_top, ranked_heads_from_stage_a, target_heads


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = build_chat_prompt(self.tokenizer, row["prompt"])
        answer = str(row["gold"])
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer + self.tokenizer.eos_token, add_special_tokens=False)[
            "input_ids"
        ]
        input_ids = (prompt_ids + answer_ids)[-self.max_length :]
        n_prompt_kept = max(0, len(prompt_ids) - max(0, len(prompt_ids) + len(answer_ids) - self.max_length))
        labels = [-100] * n_prompt_kept + input_ids[n_prompt_kept:]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_token_id):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append([pad_token_id] * pad + item["input_ids"])
        labels.append([-100] * pad + item["labels"])
        attention_mask.append([0] * pad + [1] * len(item["input_ids"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)


def configure_generic_lora(model, rank, alpha, dropout):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    return model, {"variant": "generic_peft_lora", "rank": rank, "alpha": alpha, "dropout": dropout}


def configure_adapter(model, arm, args):
    if arm == "generic":
        return configure_generic_lora(model, args.rank, args.lora_alpha, args.dropout)

    freeze_model(model)
    ranked = ranked_heads_from_stage_a(args.stage_a_npz, args.stage_a_index)
    if arm == "arm_l":
        heads = target_heads(
            args.a1_summary,
            top_k=8,
            num_layers=model.config.num_hidden_layers,
            num_heads=model.config.num_attention_heads,
        )
        variant = "head_sliced"
    elif arm == "random_heads":
        heads = random_heads_outside_top(
            ranked,
            seed=args.random_seed,
            top_exclude=32,
            k=8,
            num_layers=model.config.num_hidden_layers,
            num_heads=model.config.num_attention_heads,
        )
        variant = "random_head_sliced"
    else:
        raise ValueError(f"unknown arm: {arm}")
    config = wrap_head_sliced_adapters(model, heads, rank=args.rank, alpha=args.lora_alpha)
    config["variant"] = variant
    config["random_seed"] = args.random_seed if arm == "random_heads" else None
    config["ranked_head_source"] = {
        "npz": args.stage_a_npz,
        "index": args.stage_a_index,
        "top_exclude": 32 if arm == "random_heads" else None,
    }
    return model, config


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.train_data, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model, adapter_config = configure_adapter(model, args.arm, args)
    model.train()

    dataset = SFTDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    max_steps = args.max_steps or int(args.epochs * steps_per_epoch)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max_steps // 10),
        num_training_steps=max_steps,
    )

    losses = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_value = float(loss.detach().cpu()) * args.grad_accum
                losses.append(loss_value)
                if step == 1 or step % args.log_every == 0:
                    print(f"step={step}/{max_steps} loss={loss_value:.6f}", flush=True)
                if step >= max_steps:
                    break

    adapter_config["num_trainable"] = count_trainable_parameters(model)
    adapter_config["arm"] = args.arm
    adapter_config["model"] = args.model
    adapter_config["train_data"] = args.train_data
    adapter_config["n_train_rows"] = len(rows)
    adapter_config["max_steps"] = max_steps
    adapter_config["epochs_requested"] = args.epochs
    adapter_config["lr"] = args.lr
    adapter_config["seed"] = args.seed
    adapter_config["random_seed"] = args.random_seed
    adapter_config["losses"] = losses
    adapter_config["loss_start"] = losses[0] if losses else None
    adapter_config["loss_end"] = losses[-1] if losses else None

    if args.arm == "generic":
        model.save_pretrained(args.out_dir)
        with open(Path(args.out_dir) / "train_summary.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
    else:
        save_head_sliced_adapter(model, args.out_dir, adapter_config)
        with open(Path(args.out_dir) / "train_summary.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
    print(f"wrote adapter to {args.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["arm_l", "generic", "random_heads"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-data", default="data/stage_c/train_overwrite.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--a1-summary", default="results/stage_a/a1_stable_summary.json")
    ap.add_argument("--stage-a-npz", default="results/stage_a/extract.npz")
    ap.add_argument("--stage-a-index", default="results/stage_a/extract_index.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--random-seed", type=int, default=123)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
