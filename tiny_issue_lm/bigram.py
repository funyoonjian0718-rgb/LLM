"""The smallest useful neural language model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predict the next token using only the current token.

    Row i of ``transition_logits`` contains the unnormalized probabilities for
    the token that follows token i. This is an embedding lookup expressed as a
    neural network so that autograd and an optimizer can train it.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transition_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.transition_logits(token_ids)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        eot_id: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        for _ in range(max_new_tokens):
            logits, _ = self(token_ids[:, -1:])
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, self.vocab_size)
                cutoff = torch.topk(next_logits, k).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < cutoff, -torch.inf)
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
            if eot_id is not None and torch.all(next_token == eot_id):
                break
        return token_ids
