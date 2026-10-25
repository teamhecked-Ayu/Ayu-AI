from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("AYULM requires numpy. Install the project with `pip install -e .`.") from exc


@dataclass
class AyuLMConfig:
    vocab_size: int = 256
    model_dim: int = 32
    n_layers: int = 2
    n_heads: int = 4
    context_length: int = 64
    dropout: float = 0.0


class AyuLMTokenizer:
    def __init__(self, vocabulary: Iterable[str] | None = None):
        base = list(vocabulary) if vocabulary is not None else list(" !.,;:!?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n")
        self.vocab = list(dict.fromkeys(["<pad>", "<unk>", "<eos>"] + base))
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token = "<eos>"

    def encode(self, text: str) -> list[int]:
        tokens = []
        for character in text:
            tokens.append(self.token_to_id.get(character, self.token_to_id[self.unk_token]))
        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if token == self.eos_token:
                break
            chars.append(token)
        return "".join(chars)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]


class TinyDecoderBlock:
    def __init__(self, config: AyuLMConfig, seed: int = 0):
        self.config = config
        rng = random.Random(seed)
        self.q = np.array([[rng.uniform(-0.2, 0.2) for _ in range(config.model_dim)] for _ in range(config.model_dim)], dtype=np.float32)
        self.k = np.array([[rng.uniform(-0.2, 0.2) for _ in range(config.model_dim)] for _ in range(config.model_dim)], dtype=np.float32)
        self.v = np.array([[rng.uniform(-0.2, 0.2) for _ in range(config.model_dim)] for _ in range(config.model_dim)], dtype=np.float32)
        self.out = np.array([[rng.uniform(-0.2, 0.2) for _ in range(config.model_dim)] for _ in range(config.model_dim)], dtype=np.float32)
        self.ffn1 = np.array([[rng.uniform(-0.2, 0.2) for _ in range(config.model_dim * 2)] for _ in range(config.model_dim)], dtype=np.float32)
        self.ffn2 = np.array([[rng.uniform(-0.2, 0.2) for _ in range(config.model_dim)] for _ in range(config.model_dim * 2)], dtype=np.float32)

    def attention(self, x: np.ndarray) -> np.ndarray:
        q = x @ self.q
        k = x @ self.k
        v = x @ self.v
        scale = 1.0 / math.sqrt(self.config.model_dim)
        logits = (q @ k.T) * scale
        weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return weights @ v

    def forward(self, x: np.ndarray) -> np.ndarray:
        attn = self.attention(x)
        mixed = attn @ self.out.T
        hidden = mixed + x
        proj = hidden @ self.ffn1
        activated = np.tanh(proj)
        out = activated @ self.ffn2
        return hidden + out


class AyuLMModel:
    def __init__(self, config: AyuLMConfig | None = None, vocab_size: int | None = None, seed: int = 123):
        self.config = config or AyuLMConfig(vocab_size=vocab_size or 256)
        self.seed = seed
        self.token_embedding = np.random.default_rng(seed).normal(0.0, 0.1, size=(self.config.vocab_size, self.config.model_dim)).astype(np.float32)
        self.position_embedding = np.random.default_rng(seed + 7).normal(0.0, 0.1, size=(self.config.context_length, self.config.model_dim)).astype(np.float32)
        self.blocks = [TinyDecoderBlock(self.config, seed + index) for index in range(self.config.n_layers)]
        self.output_head = np.random.default_rng(seed + 99).normal(0.0, 0.1, size=(self.config.model_dim, self.config.vocab_size)).astype(np.float32)

    def _embed_tokens(self, token_ids: list[int]) -> np.ndarray:
        indexed = self.token_embedding[np.asarray(token_ids, dtype=np.int64)]
        positions = self.position_embedding[: indexed.shape[0]]
        return indexed + positions

    def forward(self, token_ids: list[int]) -> np.ndarray:
        x = self._embed_tokens(token_ids)
        for block in self.blocks:
            x = block.forward(x)
        logits = x @ self.output_head
        return logits

    def generate(self, token_ids: list[int], max_new_tokens: int = 8) -> list[int]:
        current = list(token_ids)
        for _ in range(max_new_tokens):
            logits = self.forward(current)
            next_id = int(np.argmax(logits[-1]))
            current.append(next_id)
            if next_id == 0:
                break
        return current

    def parameter_count(self) -> int:
        total = self.token_embedding.size + self.position_embedding.size + self.output_head.size
        for block in self.blocks:
            total += block.q.size + block.k.size + block.v.size + block.out.size + block.ffn1.size + block.ffn2.size
        return int(total)
