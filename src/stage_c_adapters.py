"""Head-sliced low-rank adapters for Stage C."""

import json
from pathlib import Path

import torch
import torch.nn as nn


class HeadSliceLoRALinear(nn.Module):
    """Low-rank update applied only to selected output rows of a Linear layer."""

    def __init__(self, base, selected_rows, rank=16, alpha=16):
        super().__init__()
        self.base = base
        self.selected_rows = tuple(int(r) for r in selected_rows)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        for param in self.base.parameters():
            param.requires_grad_(False)
        device = base.weight.device
        self.register_buffer(
            "row_index",
            torch.tensor(self.selected_rows, dtype=torch.long, device=device),
            persistent=False,
        )
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features, device=device))
        self.lora_b = nn.Parameter(torch.zeros(len(self.selected_rows), self.rank, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x):
        out = self.base(x)
        if not self.selected_rows:
            return out
        low_rank = torch.matmul(x.to(self.lora_a.dtype), self.lora_a.t())
        delta = torch.matmul(low_rank, self.lora_b.t()) * self.scaling
        out = out.clone()
        out.index_add_(-1, self.row_index.to(out.device), delta.to(out.dtype))
        return out


def q_rows_for_heads(heads, head_dim):
    rows = []
    for head in heads:
        start = int(head) * head_dim
        rows.extend(range(start, start + head_dim))
    return rows


def k_rows_for_heads(heads, head_dim, num_attention_heads, num_key_value_heads):
    groups = num_attention_heads // num_key_value_heads
    kv_groups = sorted({int(head) // groups for head in heads})
    rows = []
    for group in kv_groups:
        start = group * head_dim
        rows.extend(range(start, start + head_dim))
    return rows


def wrap_head_sliced_adapters(model, heads, rank=16, alpha=16):
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    by_layer = {}
    for item in heads:
        by_layer.setdefault(int(item["layer"]), []).append(int(item["head"]))
    wrapped = {}
    for layer, layer_heads in by_layer.items():
        attn = model.model.layers[layer].self_attn
        q_rows = q_rows_for_heads(layer_heads, head_dim)
        k_rows = k_rows_for_heads(
            layer_heads, head_dim, config.num_attention_heads, config.num_key_value_heads
        )
        attn.q_proj = HeadSliceLoRALinear(attn.q_proj, q_rows, rank=rank, alpha=alpha)
        attn.k_proj = HeadSliceLoRALinear(attn.k_proj, k_rows, rank=rank, alpha=alpha)
        wrapped[str(layer)] = {
            "heads": layer_heads,
            "q_rows": q_rows,
            "k_rows": k_rows,
        }
    return {
        "variant": "head_sliced",
        "rank": rank,
        "alpha": alpha,
        "heads": heads,
        "wrapped": wrapped,
        "num_trainable": count_trainable_parameters(model),
    }


def count_trainable_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def adapter_state_dict(model):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_a" in name or "lora_b" in name
    }


def save_head_sliced_adapter(model, out_dir, adapter_config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state_dict(model), out / "adapter_model.pt")
    with open(out / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)


def load_head_sliced_adapter(model, adapter_dir):
    adapter_dir = Path(adapter_dir)
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    wrap_head_sliced_adapters(
        model,
        config["heads"],
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
    )
    state = torch.load(adapter_dir / "adapter_model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [x for x in unexpected if "lora_" in x]
    if unexpected:
        raise RuntimeError(f"unexpected adapter keys: {unexpected}")
    return {"missing": missing, "config": config}
