"""Checkpoint loading and text generation."""

from __future__ import annotations

from pathlib import Path

import torch

from .bigram import BigramLanguageModel
from .tokenizer import CharacterTokenizer
from .training import choose_device


def load_checkpoint(
    path: str | Path, device_name: str = "auto"
) -> tuple[BigramLanguageModel, CharacterTokenizer, torch.device]:
    device = choose_device(device_name)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    tokenizer = CharacterTokenizer.from_dict(checkpoint["tokenizer"])
    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, device


def generate_text(
    checkpoint_path: str | Path,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    device_name: str = "auto",
) -> str:
    model, tokenizer, device = load_checkpoint(checkpoint_path, device_name)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.eot_id]
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        tokens,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eot_id=tokenizer.eot_id,
    )
    return tokenizer.decode(generated[0].tolist(), skip_special=True)

