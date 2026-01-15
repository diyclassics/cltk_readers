"""NLP pipeline management for latincy-readers.

This module handles lazy loading and caching of spaCy models, as well as
registration of custom extensions for citation tracking.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

import spacy
from spacy.tokens import Doc, Span, Token

if TYPE_CHECKING:
    from spacy import Language


class AnnotationLevel(Enum):
    """Control how much NLP processing to apply.

    Use lower levels for better performance when you don't need full annotations.

    Levels:
        NONE: Raw text only, no spaCy processing
        TOKENIZE: Tokenization + sentence splitting only
        BASIC: + lemmatization, POS tagging (default)
        FULL: + NER, dependency parsing
    """

    NONE = "none"
    TOKENIZE = "tokenize"
    BASIC = "basic"
    FULL = "full"


# Register custom extensions once at module load
def _register_extensions() -> None:
    """Register spaCy custom extensions for citation tracking."""
    if not Doc.has_extension("metadata"):
        Doc.set_extension("metadata", default=None)
        Doc.set_extension("fileid", default=None)

    if not Span.has_extension("citation"):
        Span.set_extension("citation", default=None)
        Span.set_extension("metadata", default=None)

    if not Token.has_extension("citation"):
        Token.set_extension("citation", default=None)


# Register on import
_register_extensions()


@lru_cache(maxsize=4)
def load_model(model_name: str = "la_core_web_lg") -> Language:
    """Load and cache a spaCy model.

    Args:
        model_name: Name of the spaCy model to load.

    Returns:
        Loaded spaCy Language pipeline.
    """
    nlp = spacy.load(model_name)
    nlp.max_length = 2_500_000  # Handle large documents
    return nlp


def create_pipeline(level: AnnotationLevel = AnnotationLevel.BASIC) -> Language | None:
    """Create a spaCy pipeline based on annotation level.

    Args:
        level: Desired annotation level.

    Returns:
        Configured spaCy pipeline, or None for NONE level.
    """
    if level == AnnotationLevel.NONE:
        return None

    if level == AnnotationLevel.TOKENIZE:
        # Minimal pipeline: just tokenization and sentence splitting
        nlp = spacy.blank("la")
        nlp.add_pipe("sentencizer")
        nlp.max_length = 2_500_000
        return nlp

    # BASIC or FULL: load the full model
    if level == AnnotationLevel.BASIC:
        # Disable heavy components we don't need
        nlp = spacy.load("la_core_web_lg", disable=["ner", "parser"])
    else:
        # FULL: everything enabled
        nlp = spacy.load("la_core_web_lg")

    nlp.max_length = 2_500_000
    return nlp


def get_nlp(level: AnnotationLevel = AnnotationLevel.BASIC) -> Language | None:
    """Get a spaCy pipeline for the given annotation level.

    This is the main entry point for getting an NLP pipeline. Pipelines are
    cached, so repeated calls with the same level are efficient.

    Args:
        level: Desired annotation level.

    Returns:
        Configured spaCy pipeline, or None for NONE level.

    Example:
        >>> nlp = get_nlp(AnnotationLevel.BASIC)
        >>> doc = nlp("Arma virumque cano.")
        >>> print([(t.text, t.lemma_) for t in doc])
    """
    return create_pipeline(level)
