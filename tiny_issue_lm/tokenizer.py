"""A deliberately small character-level tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


UNK_TOKEN = "<|unk|>"
EOT_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class CharacterTokenizer:
    """Map individual Unicode characters to integer token IDs.

    The two special tokens occupy stable IDs. All ordinary characters are
    sorted by Unicode code point, making vocabulary construction reproducible.
    """

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        if self.tokens[:2] != (UNK_TOKEN, EOT_TOKEN):
            raise ValueError("The first two tokens must be UNK and EOT")

    @classmethod
    def train(cls, documents: Iterable[str]) -> "CharacterTokenizer":
        characters = sorted(set("".join(documents)))
        return cls((UNK_TOKEN, EOT_TOKEN, *characters))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def unk_id(self) -> int:
        return 0

    @property
    def eot_id(self) -> int:
        return 1

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, text: str, *, add_eot: bool = False) -> list[int]:
        lookup = self.token_to_id
        ids = [lookup.get(character, self.unk_id) for character in text]
        if add_eot:
            ids.append(self.eot_id)
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = False) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            token = self.tokens[token_id]
            if skip_special and token in {UNK_TOKEN, EOT_TOKEN}:
                continue
            pieces.append(token)
        return "".join(pieces)

    def to_dict(self) -> dict[str, list[str]]:
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "CharacterTokenizer":
        return cls(tuple(data["tokens"]))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

