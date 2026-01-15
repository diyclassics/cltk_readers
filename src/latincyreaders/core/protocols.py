"""Protocol definitions for corpus readers.

These protocols define the interface contract that all readers should implement.
Use these for type hints when you want to accept any compatible reader.
"""

from typing import Protocol, Iterator, runtime_checkable
from pathlib import Path

# Forward references for spaCy types (avoid import at module level)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token


@runtime_checkable
class CorpusReaderProtocol(Protocol):
    """Base interface for all corpus readers.

    All readers must implement these core methods for accessing corpus data.
    """

    @property
    def root(self) -> Path:
        """Root directory of the corpus."""
        ...

    def fileids(self) -> list[str]:
        """Return list of all file identifiers in the corpus."""
        ...

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Doc objects for each document."""
        ...

    def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]:
        """Yield raw text strings for each document."""
        ...

    def sents(self, fileids: str | list[str] | None = None) -> Iterator["Span"]:
        """Yield sentence Spans from documents."""
        ...

    def tokens(self, fileids: str | list[str] | None = None) -> Iterator["Token"]:
        """Yield individual Tokens from documents."""
        ...


@runtime_checkable
class CitationReaderProtocol(CorpusReaderProtocol, Protocol):
    """Extended interface for readers that preserve citation information.

    Readers implementing this protocol track source citations (e.g., book/line
    numbers) through the processing pipeline.
    """

    def lines(self, fileids: str | list[str] | None = None) -> Iterator["Span"]:
        """Yield citation-unit Spans (lines, verses, etc.)."""
        ...

    def doc_rows(self, fileids: str | list[str] | None = None) -> Iterator[dict[str, "Span"]]:
        """Yield citation -> Span mappings for each document."""
        ...
