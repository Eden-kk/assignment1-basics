from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from os import PathLike

import regex as re


TokenId = int
TokenBytes = bytes
Merge = tuple[bytes, bytes]
Vocab = dict[TokenId, TokenBytes]


_PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@lru_cache
def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


@lru_cache
def _gpt2_unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in _gpt2_bytes_to_unicode().items()}


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Vocab, list[Merge]]:
    """
    Train a BPE tokenizer from a corpus file.

    Returns:
        vocab:
            Mapping from token id to token bytes.
        merges:
            Ordered list of learned merges.
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small for base byte vocabulary and special tokens")

    with open(input_path, mode="r", encoding="utf-8") as f:
        data = f.read()
    # initialize corpus 
    corpus: dict[tuple[bytes, ...], int] = {}
    segments = _split_by_special_tokens(data, special_tokens)
    for segment in segments:
        if segment in special_tokens:
            continue
        for pre_token in _pretokenize(segment):
            token_bytes = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
            corpus[token_bytes] = corpus.get(token_bytes, 0) + 1

    # initialize pair status
    pair_counts, pair_to_words = _initialize_pair_stats(corpus)

    # merge most frequent pairs
    merges = []
    num_merges_to_learn = vocab_size - 256 - len(special_tokens)

    for _ in range(num_merges_to_learn):
        best_pair = None if not pair_counts else max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        if not best_pair:
            break
        merges.append(best_pair)
        _apply_merge(corpus, pair_counts, pair_to_words, best_pair)

    vocab = _build_vocab_from_merges(merges, special_tokens)
    return vocab, merges


class Tokenizer:
    """
    Runtime BPE tokenizer built from a fixed vocabulary and merge list.
    """

    def __init__(
        self,
        vocab: Vocab,
        merges: list[Merge],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Construct a tokenizer from precomputed vocab and merges.
        """
        self.vocab = dict(vocab)
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens or []
        self._add_missing_special_tokens()
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_path: str | PathLike[str],
        merges_path: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        Load vocab and merges from disk and construct a tokenizer.
        """
        vocab = _load_vocab(vocab_path)
        merges = _load_merges(merges_path)
        return cls(vocab, merges, special_tokens)

    @classmethod
    def train(
        cls,
        input_path: str | PathLike[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        Optional convenience constructor.

        This should be a thin wrapper around `train_bpe(...)` plus `cls(...)`.
        """
        special_tokens = special_tokens or []
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode a full string into token ids.
        """
        segments = _split_by_special_tokens(text, self.special_tokens)
        encoded = []
        for segment in segments:
            encoded.extend(self._encode_segment(segment))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Stream-encode chunks of text from an iterable.
        """
        max_special_len = max((len(token) for token in self.special_tokens), default=0)
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            if not buffer:
                continue

            if max_special_len > 0 and len(buffer) <= max_special_len:
                continue

            safe_prefix_end = len(buffer) - _longest_special_prefix_suffix(buffer, self.special_tokens)
            safe_prefix = buffer[:safe_prefix_end]
            deferred_suffix = buffer[safe_prefix_end:]

            segments = _split_by_special_tokens(safe_prefix, self.special_tokens)
            if not segments:
                buffer = deferred_suffix
                continue

            for segment in segments[:-1]:
                yield from self._encode_segment(segment)

            last_segment = segments[-1]
            if last_segment in self.special_tokens:
                yield from self._encode_segment(last_segment)
                buffer = deferred_suffix
                continue

            previous_pretoken = None
            for pretoken in _pretokenize(last_segment):
                if previous_pretoken is not None:
                    yield from _encode_piece(
                        previous_pretoken.encode("utf-8", errors="replace"),
                        self.merge_ranks,
                        self.reversed_vocab,
                    )
                previous_pretoken = pretoken

            buffer = (previous_pretoken or "") + deferred_suffix

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        """
        Decode token ids back into a string.
        """
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def decode_bytes(self, ids: list[int]) -> bytes:
        """
        Decode token ids back into raw bytes before UTF-8 conversion.
        Optional but often useful internally.
        """
        return b"".join(self.vocab[token_id] for token_id in ids)

    def token_to_id(self, token: bytes) -> int:
        """
        Look up the id for a token in the vocabulary.
        """
        return self.reversed_vocab[token]

    def id_to_token(self, token_id: int) -> bytes:
        """
        Look up the token bytes for a token id.
        """
        return self.vocab[token_id]

    def _encode_segment(self, segment: str) -> list[int]:
        if segment in self.special_tokens:
            return [self.token_to_id(segment.encode("utf-8", errors="replace"))]

        encoded = []
        for pre_token in _pretokenize(segment):
            encoded.extend(
                _encode_piece(
                    pre_token.encode("utf-8", errors="replace"),
                    self.merge_ranks,
                    self.reversed_vocab,
                )
            )
        return encoded

    def _add_missing_special_tokens(self) -> None:
        next_token_id = max(self.vocab.keys(), default=-1) + 1
        existing_tokens = set(self.vocab.values())
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in existing_tokens:
                self.vocab[next_token_id] = special_token_bytes
                existing_tokens.add(special_token_bytes)
                next_token_id += 1


def _load_vocab(vocab_path: str | PathLike[str]) -> Vocab:
    """Read vocab from disk."""
    with open(vocab_path, encoding="utf-8") as f:
        serialized_vocab = json.load(f)

    gpt2_byte_decoder = _gpt2_unicode_to_bytes()
    return {
        int(token_id): bytes(gpt2_byte_decoder[ch] for ch in token_string)
        for token_string, token_id in serialized_vocab.items()
    }


def _load_merges(merges_path: str | PathLike[str]) -> list[Merge]:
    """Read merges from disk."""
    merges = []
    gpt2_byte_decoder = _gpt2_unicode_to_bytes()
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.rstrip()
            if not cleaned_line:
                continue
            parts = cleaned_line.split(" ")
            if len(parts) != 2:
                continue
            left, right = parts
            merges.append(
                (
                    bytes(gpt2_byte_decoder[ch] for ch in left),
                    bytes(gpt2_byte_decoder[ch] for ch in right),
                )
            )
    return merges


def _split_by_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
    """
    Split text so special tokens are preserved as atomic segments.
    """
    if not special_tokens:
        return [text]

    escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    pattern = re.compile(f"({'|'.join(escaped_tokens)})")
    return [part for part in pattern.split(text) if part]


def _pretokenize(text: str) -> Iterator[str]:
    """
    Yield pretokenized string pieces before BPE merging.
    """
    for match in _PRETOKENIZE_PATTERN.finditer(text):
        yield match.group(0)


def _encode_piece(
    piece: bytes,
    merge_ranks: dict[Merge, int],
    token_to_id_map: dict[bytes, int],
) -> list[int]:
    """
    Encode one non-special byte piece with learned BPE merges.
    """
    if piece in token_to_id_map:
        return [token_to_id_map[piece]]
        
    tokens = [bytes([b]) for b in piece]
    while len(tokens) > 1:
        best_rank = None
        best_pair = None
        for pair in zip(tokens[:-1], tokens[1:]):
            if pair in merge_ranks:
                rank = merge_ranks[pair]
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

        if best_pair is None:
            break

        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        tokens = merged_tokens
    
    return [token_to_id_map[token] for token in tokens]


def _pairs_in_word(word: tuple[bytes, ...]) -> dict[Merge, int]:
    pair_counts = dict()
    for pair in zip(word[:-1], word[1:]):
        if not pair in pair_counts:
            pair_counts[pair] = 0
        pair_counts[pair] += 1
    return pair_counts


def _merge_pair_in_word(word: tuple[bytes, ...], pair: Merge) -> tuple[bytes, ...]:
    merged = list()
    i = 0
    while i < len(word):
        if i+1 < len(word) and (word[i], word[i+1]) == pair:
            merged.append(word[i]+ word[i+1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def _initialize_pair_stats(
    word_counts: dict[tuple[bytes, ...], int],
) -> tuple[dict[Merge, int], dict[Merge, set[tuple[bytes, ...]]]]:
    pair_counts, pair_to_words = dict(), dict()
    for word, count in word_counts.items():
        for pair, pair_count in _pairs_in_word(word).items():
            if not pair in pair_counts:
                pair_counts[pair] = 0
            pair_counts[pair] += count * pair_count
            if not pair in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)
    return pair_counts, pair_to_words


def _apply_merge(
    word_counts: dict[tuple[bytes, ...], int],
    pair_counts: dict[Merge, int],
    pair_to_words: dict[Merge, set[tuple[bytes, ...]]],
    pair: Merge,
) -> None:
    words_to_merge = list(pair_to_words[pair])
    for word_to_merge in words_to_merge:
        count = word_counts[word_to_merge]
        merged_word = _merge_pair_in_word(word_to_merge, pair)

        word_counts.pop(word_to_merge)
        if not merged_word in word_counts:
            word_counts[merged_word] = 0
        word_counts[merged_word] += count

        old_pair_counts = _pairs_in_word(word_to_merge)
        new_pair_counts = _pairs_in_word(merged_word)

        for old_pair, pair_count in old_pair_counts.items():
            pair_counts[old_pair] -= count * pair_count
            if pair_counts[old_pair] == 0:
                pair_counts.pop(old_pair)
            pair_to_words[old_pair].remove(word_to_merge)
        
        for new_pair, pair_count in new_pair_counts.items():
            if not new_pair in pair_counts:
                pair_counts[new_pair] = 0
            pair_counts[new_pair] += count * pair_count
            if not new_pair in pair_to_words:
                pair_to_words[new_pair] = set()
            pair_to_words[new_pair].add(merged_word)


def _build_vocab_from_merges(
    merges: list[Merge],
    special_tokens: list[str],
) -> Vocab:
    """
    Build final vocab from base byte vocabulary, learned merges, and special tokens.
    """
    vocab = dict[TokenId, TokenBytes]()
    current_token_id: TokenId = 0
    # special tokens
    for special_token in special_tokens:
        vocab[current_token_id] = special_token.encode("utf-8")
        current_token_id += 1

    # base tokens: 256 single-byte tokens
    for b in range(256):
        vocab[current_token_id] = bytes([b])
        current_token_id += 1

    # merged tokens
    for left, right in merges:
        vocab[current_token_id] = left + right
        current_token_id += 1

    return vocab


def _longest_special_prefix_suffix(
    buffer: str,
    special_tokens: list[str],
) -> int:
    """
    Return the longest suffix of `buffer` that is a strict prefix of a special token.
    """
    longest = 0
    for token in special_tokens:
        max_check = min(len(buffer), len(token) - 1)
        for k in range(1, max_check + 1):
            if buffer.endswith(token[:k]):
                longest = max(longest, k)
    return longest