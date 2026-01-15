"""Plaintext corpus readers.

Readers for plain text files without specialized markup. Includes:
- PlaintextReader: Generic plaintext reader
- LatinLibraryReader: Reader for The Latin Library corpus
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


class PlaintextReader(BaseCorpusReader):
    """Reader for plain text Latin files.

    Handles generic plaintext files with paragraph-based structure.
    Paragraphs are separated by blank lines.

    Example:
        >>> reader = PlaintextReader("/path/to/texts")
        >>> for doc in reader.docs():
        ...     for sent in doc.sents:
        ...         print(sent.text)
    """

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default to .txt files."""
        return "**/*.txt"

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a plaintext file.

        Yields the entire file content as a single text chunk.

        Args:
            path: Path to text file.

        Yields:
            Single (text, metadata) tuple per file.
        """
        text = path.read_text(encoding=self._encoding)
        text = self._normalize_text(text)

        if not text.strip():
            return

        metadata = {
            "filename": path.name,
            "path": str(path),
        }

        yield text, metadata

    def paras(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield paragraphs from documents.

        Paragraphs are identified by blank line separation in the source text.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Span objects.

        Yields:
            Paragraph Spans (or strings if as_text=True).
        """
        for path in self._iter_paths(fileids):
            text = path.read_text(encoding=self._encoding)
            text = self._normalize_text(text)

            # Split on blank lines
            para_texts = [p.strip() for p in text.split("\n\n") if p.strip()]

            if as_text:
                yield from para_texts
            else:
                # Need NLP for spans
                nlp = self.nlp
                if nlp is None:
                    raise ValueError(
                        "Cannot create paragraph Spans with annotation_level=NONE. "
                        "Use paras(as_text=True) or set a higher annotation level."
                    )
                for para_text in para_texts:
                    doc = nlp(para_text)
                    # Yield the whole doc as a span
                    yield doc[:]


class LatinLibraryReader(PlaintextReader):
    """Reader for The Latin Library corpus.

    The Latin Library (https://www.thelatinlibrary.com/) is a collection
    of Latin texts in plain text format. This reader handles the standard
    structure of Latin Library files.

    If no root is provided, attempts to use the CLTK-downloaded corpus.

    Example:
        >>> reader = LatinLibraryReader()  # Uses CLTK default location
        >>> for doc in reader.docs():
        ...     print(f"{doc._.fileid}: {len(list(doc.sents))} sentences")
    """

    # Default CLTK corpus location
    CLTK_CORPUS_NAME = "lat_text_latin_library"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.BASIC,
        auto_download: bool = False,
    ):
        """Initialize the Latin Library reader.

        Args:
            root: Root directory. If None, uses CLTK default location.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            auto_download: If True and corpus not found, download via CLTK.
        """
        if root is None:
            root = self._get_default_root(auto_download)

        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
        )

    def _get_default_root(self, auto_download: bool = False) -> Path:
        """Get the default CLTK corpus location.

        Args:
            auto_download: If True, download corpus if not present.

        Returns:
            Path to corpus root.

        Raises:
            FileNotFoundError: If corpus not found and auto_download=False.
        """
        from pathlib import Path
        import os

        # Standard CLTK data location
        cltk_data = Path(os.path.expanduser("~/cltk_data"))
        corpus_path = cltk_data / "lat" / "text" / self.CLTK_CORPUS_NAME

        if corpus_path.exists():
            return corpus_path

        if auto_download:
            try:
                from cltk.data.fetch import FetchCorpus
                fetcher = FetchCorpus(language="lat")
                fetcher.import_corpus(self.CLTK_CORPUS_NAME)
                return corpus_path
            except ImportError:
                raise ImportError(
                    "CLTK is required for auto-download. "
                    "Install with: pip install cltk"
                )

        raise FileNotFoundError(
            f"Latin Library corpus not found at {corpus_path}. "
            f"Either provide a root path or set auto_download=True."
        )

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a Latin Library file.

        Latin Library files may have headers/footers that should be cleaned.

        Args:
            path: Path to text file.

        Yields:
            Single (text, metadata) tuple per file.
        """
        text = path.read_text(encoding=self._encoding)
        text = self._normalize_text(text)
        text = self._clean_latin_library_text(text)

        if not text.strip():
            return

        # Extract title from filename
        title = path.stem.replace("_", " ").title()

        metadata = {
            "filename": path.name,
            "path": str(path),
            "title": title,
        }

        yield text, metadata

    def _clean_latin_library_text(self, text: str) -> str:
        """Clean Latin Library-specific formatting.

        Args:
            text: Raw text from file.

        Returns:
            Cleaned text.
        """
        # Latin Library texts are generally clean
        # This method can be extended for specific cleanup needs
        lines = text.split("\n")

        # Remove common header/footer patterns if present
        cleaned_lines = []
        for line in lines:
            # Skip typical navigation lines
            if line.strip().lower() in ("the latin library", "home", ""):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)
