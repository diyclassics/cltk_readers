"""Base corpus reader class.

This module provides the abstract base class that all corpus readers inherit from.
It handles common functionality like file discovery, NLP pipeline management,
and the standard iteration interface.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from latincyreaders.nlp.pipeline import AnnotationLevel, get_nlp

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Doc, Span, Token

# Re-export for convenience
__all__ = ["BaseCorpusReader", "AnnotationLevel"]


class BaseCorpusReader(ABC):
    """Abstract base class for all Latin corpus readers.

    To create a new reader, subclass and implement:

    Required:
        - _parse_file(path) -> yields (text, metadata) tuples

    Optional overrides:
        - _normalize_text(text) -> cleaned text
        - _default_file_pattern() -> glob pattern for files

    Example:
        class MyReader(BaseCorpusReader):
            @classmethod
            def _default_file_pattern(cls) -> str:
                return "*.txt"

            def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
                yield path.read_text(), {"filename": path.name}
    """

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.BASIC,
    ):
        """Initialize the corpus reader.

        Args:
            root: Root directory containing corpus files.
            fileids: Glob pattern for selecting files. If None, uses class default.
            encoding: Text encoding for reading files.
            annotation_level: How much NLP annotation to apply.
        """
        self._root = Path(root).resolve()
        self._fileids_pattern = fileids or self._default_file_pattern()
        self._encoding = encoding
        self._annotation_level = annotation_level
        self._nlp: Language | None = None  # Lazy loaded

    @property
    def root(self) -> Path:
        """Root directory of the corpus."""
        return self._root

    @property
    def nlp(self) -> Language | None:
        """spaCy pipeline (lazy loaded on first access)."""
        if self._nlp is None and self._annotation_level != AnnotationLevel.NONE:
            self._nlp = get_nlp(self._annotation_level)
        return self._nlp

    @property
    def annotation_level(self) -> AnnotationLevel:
        """Current annotation level."""
        return self._annotation_level

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default glob pattern for this corpus type. Override in subclasses."""
        return "*.*"

    def _normalize_text(self, text: str) -> str:
        """Normalize text. Override for corpus-specific cleaning.

        Args:
            text: Raw text from file.

        Returns:
            Normalized text.
        """
        return unicodedata.normalize("NFC", text)

    def fileids(self, match: str | None = None) -> list[str]:
        """Return list of file identifiers matching the pattern.

        Args:
            match: Optional regex pattern to filter filenames.

        Returns:
            Sorted list of matching file identifiers (relative paths).
        """
        pattern = self._fileids_pattern
        files = sorted(self._root.glob(pattern))

        # Convert to relative paths as strings
        result = [str(f.relative_to(self._root)) for f in files if f.is_file()]

        # Apply optional regex filter
        if match:
            regex = re.compile(match, re.IGNORECASE)
            result = [f for f in result if regex.search(f)]

        return result

    def _resolve_fileids(self, fileids: str | list[str] | None) -> list[str]:
        """Resolve fileids argument to a list of file identifiers.

        Args:
            fileids: Single fileid, list of fileids, or None for all files.

        Returns:
            List of file identifiers.
        """
        if fileids is None:
            return self.fileids()
        if isinstance(fileids, str):
            return [fileids]
        return list(fileids)

    def _iter_paths(self, fileids: str | list[str] | None = None) -> Iterator[Path]:
        """Iterate over file paths for the given fileids.

        Args:
            fileids: Files to iterate over.

        Yields:
            Path objects for each file.
        """
        for fid in self._resolve_fileids(fileids):
            yield self._root / fid

    @abstractmethod
    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a single file. Yield (text_chunk, metadata) pairs.

        This is the main extension point for subclasses. Implement this method
        to handle the specific file format of your corpus.

        Args:
            path: Path to the file to parse.

        Yields:
            Tuples of (text, metadata_dict) for each logical unit in the file.
        """
        ...

    # -------------------------------------------------------------------------
    # Core iteration methods
    # -------------------------------------------------------------------------

    def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]:
        """Yield raw text strings. Zero NLP overhead.

        This is the fastest way to iterate over corpus content when you
        don't need any linguistic annotation.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Raw text strings.
        """
        for path in self._iter_paths(fileids):
            for text, _metadata in self._parse_file(path):
                yield self._normalize_text(text)

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Doc objects with annotations.

        The level of annotation depends on the reader's annotation_level setting.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot create Docs with annotation_level=NONE. "
                "Use texts() for raw strings, or set a higher annotation level."
            )

        for path in self._iter_paths(fileids):
            for text, metadata in self._parse_file(path):
                text = self._normalize_text(text)
                doc = nlp(text)
                doc._.fileid = str(path.relative_to(self._root))
                doc._.metadata = metadata
                yield doc

    def sents(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield sentences from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Span objects.

        Yields:
            Sentence Spans (or strings if as_text=True).
        """
        for doc in self.docs(fileids):
            for sent in doc.sents:
                yield sent.text if as_text else sent

    def tokens(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Token | str"]:
        """Yield individual tokens from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Token objects.

        Yields:
            Tokens (or strings if as_text=True).
        """
        for doc in self.docs(fileids):
            for token in doc:
                yield token.text if as_text else token
