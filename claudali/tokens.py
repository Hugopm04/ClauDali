"""Counting CLIP tokens, and packing a prompt into the chunks CLIP actually sees.

SDXL's text encoders read 77 tokens at a time: two of those are the start and
end markers, so 75 carry content. compel does not truncate past that -- it
encodes chunk after chunk and concatenates the embeddings -- which is the only
reason a 300-token prompt works at all.

What it cannot do is make a later chunk mention the subject. Cross-attention in
the UNet is a softmax over the whole concatenated sequence, so a five-chunk
prompt whose subject appears only in the first chunk spends four fifths of its
conditioning describing a scene with nothing in it. That was a real, measured
failure: a forest with fairies compiled to 306 tokens, the word "fairies"
landed at tokens 46, 58 and 73, and the rendered images were empty forests.

This module exists so the compiler can see chunk boundaries before the GPU
does. It is stdlib-only at import time; the tokenizer is loaded lazily and the
count degrades to an estimate rather than failing, because everything up to the
sampler is meant to work with no weights and no network.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any, Optional

# CLIP's context is 77 positions, two of which are the BOS and EOS markers.
CHUNK_TOKENS = 77
CHUNK_CONTENT_TOKENS = CHUNK_TOKENS - 2

# One run of letters, one run of digits, or one other visible character. CLIP's
# BPE splits on exactly these boundaries before merging pairs, so counting them
# separately is what makes the estimate track the real tokenizer.
_PIECE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


@dataclass(frozen=True)
class TokenCount:
    """How long a prompt is, and how much to trust the number."""

    tokens: int
    chunks: int
    exact: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "chunks": self.chunks,
            "exact": self.exact,
            "source": self.source,
        }


def chunks_for(content_tokens: int) -> int:
    """How many 77-token windows a given number of content tokens occupies."""
    if content_tokens <= 0:
        return 0
    return -(-content_tokens // CHUNK_CONTENT_TOKENS)


def estimate_tokens(text: str) -> int:
    """Approximate CLIP's token count without loading a tokenizer.

    The measure this replaces was ``len(text.split()) * 1.3``, which read 212
    for a prompt the real tokenizer made 306 -- low by 31%, enough that the
    over-length warning never fired on the prompt that needed it. Splitting on
    whitespace cannot work here, because a compiled prompt is mostly commas and
    ``(phrase)1.15`` weight syntax: every bracket, comma and digit run is a
    token of its own and none of them is a word.

    A run of letters costs one token, plus one more for every six characters
    past the third: BPE has a whole-word merge for common short words and runs
    out of them on the long tail. Digits pair up. Every other visible character
    is a token of its own.

    The two constants are fitted, not chosen. Measured against CLIP's own
    tokenizer over fourteen real compiled prompts -- every example spec's
    positive and negative, plus the fairy prompt that exposed the bug -- this
    lands within 3.7% at worst and 1.5% on average. Re-fit it if the vocabulary
    tables change character a lot; do not tune it by eye.
    """
    total = 0
    for piece in _PIECE.findall(text):
        head = piece[0]
        if head.isalpha():
            total += 1 + max(0, len(piece) - 3) // 6
        elif head.isdigit():
            total += 1 + (len(piece) - 1) // 2
        else:
            total += 1
    return total


@functools.lru_cache(maxsize=1)
def _tokenizer() -> Optional[Any]:
    """A CLIP tokenizer from whichever model happens to be installed, or None.

    Deliberately opportunistic. The tokenizer files ship inside every SDXL
    checkpoint, so if the machine can render at all it can count exactly; if no
    weights have been downloaded yet the compiler still has to work, so this
    returns None rather than raising. transformers is imported here and not at
    module scope to keep ``claudali.compiler`` cheap to import.
    """
    try:
        from transformers import CLIPTokenizerFast  # noqa: F401
    except Exception:  # noqa: BLE001 - no transformers is a normal state here
        return None

    from .config import WEIGHTS_DIR

    for candidate in sorted(WEIGHTS_DIR.glob("*/tokenizer")):
        if not (candidate / "vocab.json").exists():
            continue
        try:
            from transformers import CLIPTokenizerFast

            tokenizer = CLIPTokenizerFast.from_pretrained(str(candidate))
            # Counting past 77 tokens is the entire point of this module, and
            # transformers prints a warning about indexing errors every time it
            # happens. Nothing is ever fed to a model from here, so raising the
            # declared maximum silences a warning that is wrong in this context.
            tokenizer.model_max_length = 10**9
            return tokenizer
        except Exception:  # noqa: BLE001 - a broken checkout must not break compiling
            continue
    return None


def count_content_tokens(text: str) -> tuple[int, bool, str]:
    """Content tokens in ``text``, excluding the BOS/EOS markers.

    Returns the count, whether it is exact, and what produced it. The caller
    reports all three, because a number that is sometimes measured and
    sometimes guessed is misleading unless it says which it is.
    """
    if not text.strip():
        return 0, True, "empty"
    tokenizer = _tokenizer()
    if tokenizer is None:
        return estimate_tokens(text), False, "estimate"
    try:
        ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    except Exception:  # noqa: BLE001 - fall back rather than fail a compile
        return estimate_tokens(text), False, "estimate"
    return max(0, len(ids) - 2), True, "clip-tokenizer"


def count(text: str) -> TokenCount:
    """Measure ``text`` and say how many CLIP chunks it will occupy."""
    tokens, exact, source = count_content_tokens(text)
    return TokenCount(tokens=tokens, chunks=chunks_for(tokens), exact=exact, source=source)


def pack(pieces: list[str], anchor: str = "", budget: int = CHUNK_CONTENT_TOKENS) -> list[list[str]]:
    """Group ``pieces`` into chunks of at most ``budget`` tokens each.

    ``anchor`` is reserved out of every chunk after the first but is not
    inserted here; the caller decides what to do with the grouping. Reserving it
    is what makes the guarantee below hold.

    **Why a greedy pack is enough.** compel slices at fixed 75-token positions,
    not at comma boundaries, so these chunks will not line up with its windows
    exactly. They do not need to. If an anchor is emitted at the head of every
    chunk and no chunk exceeds 75 tokens, then consecutive anchors are at most
    75 tokens apart, so any window of 75 consecutive tokens must contain one --
    otherwise two neighbouring anchors would straddle a 75-token gap, which the
    packing forbids. Every chunk CLIP sees therefore mentions the subject,
    whatever offset compel starts at.

    The one way to break that is a single piece longer than a whole chunk. It is
    placed alone and the caller is expected to warn, because nothing can be done
    about it here without dropping text the caller asked for.
    """
    anchor_cost = count_content_tokens(anchor)[0] + 1 if anchor.strip() else 0
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0
    for piece in pieces:
        cost = count_content_tokens(piece)[0] + (1 if current else 0)
        limit = budget if not chunks else budget - anchor_cost
        if current and used + cost > limit:
            chunks.append(current)
            current = [piece]
            used = count_content_tokens(piece)[0]
            continue
        current.append(piece)
        used += cost
    if current:
        chunks.append(current)
    return chunks


__all__ = [
    "CHUNK_TOKENS",
    "CHUNK_CONTENT_TOKENS",
    "TokenCount",
    "chunks_for",
    "count",
    "count_content_tokens",
    "estimate_tokens",
    "pack",
]
