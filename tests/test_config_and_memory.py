from __future__ import annotations

from ayu.ayulm import AyuLMConfig, AyuLMModel, AyuLMTokenizer


def test_tokenizer_round_trip() -> None:
    tokenizer = AyuLMTokenizer()
    text = "AYU AI"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decode_only_text(decoded) == decode_only_text(text)


def decode_only_text(value: str) -> str:
    return value.replace("<unk>", "").replace("<pad>", "").replace("<eos>", "")


def test_model_parameter_count() -> None:
    tokenizer = AyuLMTokenizer()
    model = AyuLMModel(AyuLMConfig(vocab_size=len(tokenizer.vocab), model_dim=16, n_layers=1, n_heads=2, context_length=32), vocab_size=len(tokenizer.vocab))
    assert model.parameter_count() > 0
    logits = model.forward(tokenizer.encode("ayu"))
    assert logits.shape[-1] == len(tokenizer.vocab)


def test_generate_length() -> None:
    tokenizer = AyuLMTokenizer()
    model = AyuLMModel(AyuLMConfig(vocab_size=len(tokenizer.vocab), model_dim=16, n_layers=1, n_heads=2, context_length=32), vocab_size=len(tokenizer.vocab))
    output = model.generate(tokenizer.encode("aid"), max_new_tokens=4)
    assert len(output) >= 4
