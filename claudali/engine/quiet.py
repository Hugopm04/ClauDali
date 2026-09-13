"""Three library warnings that are wrong in ClauDali's context, and nothing else.

Each was traced to the line that emits it and is silenced by matching its text
on the one logger that logs it, so every other message from those libraries
still prints. Lowering a logger's level instead would have hidden the next real
warning along with these.

1. ``diffusers.models.modeling_utils``: *"There are modules in
   UNet2DConditionModel that should be kept in float32: []"*. diffusers tests
   ``fp32_modules is not None`` on a value that is never None, so it fires on
   every dtype cast, and the empty list is the proof there was nothing to keep.
   Dropped only when the list is empty; a real list still prints.
2. ``transformers.utils.import_utils``: *"`Siglip2ImageProcessorFast` is
   deprecated"*. diffusers 0.40 imports every pipeline it ships, and its
   Z-Image pipeline still uses a class name transformers 5 renamed. No ClauDali
   code path touches it.
3. ``transformers.tokenization_utils_base``: *"Token indices sequence length is
   longer than the specified maximum sequence length (131 > 77)"*. compel
   tokenizes a whole prompt in order to split it into 77-token windows itself --
   ``CompelForSDXL`` builds both of its providers with
   ``truncate_long_prompts=False`` -- so nothing over 77 tokens ever reaches a
   text encoder and the promised indexing error cannot happen. Dropped only
   while compel is encoding, because the same message from anywhere else would
   be true.

Filters sit on the named loggers rather than on handlers. A logger's filters
run before its handlers and before propagation, and both libraries hand out
plain stdlib loggers, so this works whether or not either library has set up
its own logging yet.

Stdlib only: installing the filters must not import either library.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator

_FLOAT32_LOGGER = "diffusers.models.modeling_utils"
_SIGLIP_LOGGER = "transformers.utils.import_utils"
_TOKENS_LOGGER = "transformers.tokenization_utils_base"


class _Drop(logging.Filter):
    """Drop a record whose message contains every one of ``fragments``."""

    def __init__(self, *fragments: str) -> None:
        super().__init__()
        self.fragments = fragments

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not all(fragment in message for fragment in self.fragments)


_EMPTY_FLOAT32 = _Drop("that should be kept in float32: []")
_SIGLIP = _Drop("`Siglip2ImageProcessorFast` is deprecated")
_LONG_SEQUENCE = _Drop("Token indices sequence length is longer than the specified maximum")


def install() -> None:
    """Attach the two process-wide filters. Idempotent."""
    for name, rule in ((_FLOAT32_LOGGER, _EMPTY_FLOAT32), (_SIGLIP_LOGGER, _SIGLIP)):
        logger = logging.getLogger(name)
        if rule not in logger.filters:
            logger.addFilter(rule)


@contextlib.contextmanager
def compel_tokenization() -> Iterator[None]:
    """Silence the over-length tokenizer warning while compel encodes a prompt."""
    logger = logging.getLogger(_TOKENS_LOGGER)
    if _LONG_SEQUENCE in logger.filters:
        # Already inside a compel call; the outer one removes the filter.
        yield
        return
    logger.addFilter(_LONG_SEQUENCE)
    try:
        yield
    finally:
        logger.removeFilter(_LONG_SEQUENCE)


__all__ = ["compel_tokenization", "install"]
