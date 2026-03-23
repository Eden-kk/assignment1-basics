from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .tokenizer import Tokenizer


@dataclass(slots=True)
class PrepareDataConfig:
    input_path: Path = Path("data/TinyStoriesV2-GPT4-train.txt")
    output_path: Path = Path("data/TinyStoriesV2-GPT4-train.npy")
    vocab_path: Path = Path("assets/tokenizer/gpt2_vocab.json")
    merges_path: Path = Path("assets/tokenizer/gpt2_merges.txt")
    special_tokens: tuple[str, ...] = ("<|endoftext|>",)
    output_dtype: str = "int32"
    chunk_size: int = 1 << 20


DEFAULT_CONFIG = PrepareDataConfig()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build an argument parser for overriding data-preparation defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_CONFIG.input_path)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_CONFIG.output_path)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_CONFIG.vocab_path)
    parser.add_argument("--merges-path", type=Path, default=DEFAULT_CONFIG.merges_path)
    parser.add_argument("--output-dtype", type=str, default=DEFAULT_CONFIG.output_dtype)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CONFIG.chunk_size)
    parser.add_argument("--special-token", dest="special_tokens", action="append")
    return parser


def parse_config(argv: Sequence[str] | None = None) -> PrepareDataConfig:
    """Parse command-line overrides and return the resulting data-preparation config."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = PrepareDataConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=tuple(args.special_tokens) if args.special_tokens is not None else DEFAULT_CONFIG.special_tokens,
        output_dtype=args.output_dtype,
        chunk_size=args.chunk_size,
    )
    return config


def load_tokenizer(config: PrepareDataConfig) -> Tokenizer:
    """Load the tokenizer that will be used to encode the raw text corpus."""
    return Tokenizer.from_files(
        vocab_path=config.vocab_path, 
        merges_path=config.merges_path,
        special_tokens=config.special_tokens,
    )


def iter_text_chunks(config: PrepareDataConfig) -> Iterator[str]:
    with open(config.input_path, mode="r", encoding="utf-8") as f:
        while True:
            chunk = f.read(config.chunk_size)
            if chunk == "":
                break
            yield chunk


def encode_corpus(tokenizer: Tokenizer, text_chunks: Iterator[str]) -> Iterator[int]:
    """Encode the raw text corpus into a 1D array of token ids."""
    return tokenizer.encode_iterable(text_chunks)


def save_token_ids(token_ids: Iterator[int], config: PrepareDataConfig) -> None:
    """Persist encoded token ids to disk."""
    dtype = np.dtype(config.output_dtype)

    buffer: list[int] = []
    chunks: list[np.ndarray] = []

    for token_id in token_ids:
        buffer.append(token_id)
        if len(buffer) >= config.chunk_size:
            chunks.append(np.array(buffer, dtype=dtype))
            buffer.clear()
    
    if buffer:
        chunks.append(np.array(buffer, dtype=dtype))

    if chunks:
        token_array = np.concatenate(chunks)
    else:
        token_array = np.array([], dtype=dtype)

    np.save(config.output_path, token_array)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for tokenizing a raw text corpus into token ids for training."""
    config = parse_config(argv)
    tokenizer = load_tokenizer(config)
    chunks = iter_text_chunks(config)
    token_ids = encode_corpus(tokenizer, chunks)
    save_token_ids(token_ids, config)


if __name__ == "__main__":
    main()
