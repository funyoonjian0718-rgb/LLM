"""Data preparation and training for the bigram milestone."""

from __future__ import annotations

import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .bigram import BigramLanguageModel
from .tokenizer import CharacterTokenizer


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    batch_size: int = 32
    context_length: int = 64
    learning_rate: float = 0.05
    eval_interval: int = 100
    eval_batches: int = 20
    validation_fraction: float = 0.1
    seed: int = 1337
    device: str = "auto"


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot use it")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_documents(text: str) -> list[str]:
    """Treat blank-line-separated blocks as documents."""
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def make_document_split(
    documents: list[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    if len(documents) < 2:
        raise ValueError(
            "Training text needs at least two blank-line-separated documents "
            "so validation data can be held out before tokenization"
        )
    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(documents) * validation_fraction))
    validation_count = min(validation_count, len(documents) - 1)
    validation_indices = set(indices[:validation_count])
    train = [doc for index, doc in enumerate(documents) if index not in validation_indices]
    validation = [doc for index, doc in enumerate(documents) if index in validation_indices]
    return train, validation


def encode_documents(
    documents: list[str], tokenizer: CharacterTokenizer
) -> torch.Tensor:
    ids: list[int] = []
    for document in documents:
        ids.extend(tokenizer.encode(document, add_eot=True))
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= context_length:
        raise ValueError(
            f"Token split has {len(data)} tokens but context_length is {context_length}"
        )
    starts = torch.randint(0, len(data) - context_length, (batch_size,))
    inputs = torch.stack([data[i : i + context_length] for i in starts])
    targets = torch.stack([data[i + 1 : i + context_length + 1] for i in starts])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for name, data in (("train", train_tokens), ("validation", validation_tokens)):
        values = []
        for _ in range(config.eval_batches):
            inputs, targets = sample_batch(
                data, config.batch_size, config.context_length, device
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            values.append(loss.item())
        losses[name] = sum(values) / len(values)
    model.train()
    return losses


def save_checkpoint(
    path: str | Path,
    model: BigramLanguageModel,
    tokenizer: CharacterTokenizer,
    config: TrainConfig,
    step: int,
    metrics: dict[str, float],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_type": "bigram",
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "config": asdict(config),
            "step": step,
            "metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, destination)


def train_bigram(
    text: str, checkpoint_path: str | Path, config: TrainConfig
) -> dict[str, object]:
    seed_everything(config.seed)
    documents = split_documents(text)
    train_documents, validation_documents = make_document_split(
        documents, config.validation_fraction, config.seed
    )
    tokenizer = CharacterTokenizer.train(train_documents)
    train_tokens = encode_documents(train_documents, tokenizer)
    validation_tokens = encode_documents(validation_documents, tokenizer)
    device = choose_device(config.device)
    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    metrics: dict[str, float] = {}
    for step in range(config.steps + 1):
        if step % config.eval_interval == 0 or step == config.steps:
            metrics = estimate_loss(
                model, train_tokens, validation_tokens, config, device
            )
            print(
                f"step {step:5d} | train {metrics['train']:.4f} "
                f"| validation {metrics['validation']:.4f}"
            )
        if step == config.steps:
            break
        inputs, targets = sample_batch(
            train_tokens, config.batch_size, config.context_length, device
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    save_checkpoint(
        checkpoint_path, model, tokenizer, config, config.steps, metrics
    )
    return {
        "device": str(device),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "metrics": metrics,
    }

