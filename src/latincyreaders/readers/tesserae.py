"""Tesserae corpus reader.

Reads texts in the Tesserae format where each line contains a citation
followed by text:

    <cic. amicit. 1> Q. Mucius augur multa narrare...
    <cic. amicit. 2> cum saepe multa, tum memini...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


@dataclass
class TesseraeLine:
    """A single citation-text pair from a Tesserae file."""

    citation: str
    text: str
    line_number: int


class TesseraeReader(BaseCorpusReader):
    """Reader for Tesserae-format Latin texts with citation preservation.

    The Tesserae format uses angle-bracket citations at the start of each line:

        <author. work. section> Text content here...

    This reader preserves citation information through the NLP pipeline by
    storing it in spaCy custom extensions.

    Example:
        >>> reader = TesseraeReader("/path/to/corpus")
        >>> for doc in reader.docs():
        ...     for line in doc.spans["lines"]:
        ...         print(f"{line._.citation}: {line.text[:50]}...")

    Attributes:
        CITATION_PATTERN: Regex for parsing citation-text pairs.
    """

    CITATION_PATTERN = re.compile(r"<([^>]+)>\s*(.+)")

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Tesserae files use .tess extension."""
        return "**/*.tess"

    def _parse_lines(self, text: str) -> Iterator[TesseraeLine]:
        """Parse citation-text pairs from Tesserae format.

        Handles line continuations (lines not starting with '<' are
        appended to the previous line).

        Args:
            text: Raw file content.

        Yields:
            TesseraeLine objects for each citation unit.
        """
        current_citation: str | None = None
        current_text: str | None = None
        line_num = 0

        for raw_line in text.strip().split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            match = self.CITATION_PATTERN.match(raw_line)
            if match:
                # New citation - yield previous if exists
                if current_citation is not None and current_text is not None:
                    yield TesseraeLine(
                        citation=f"<{current_citation}>",
                        text=current_text.strip(),
                        line_number=line_num,
                    )
                    line_num += 1

                current_citation = match.group(1)
                current_text = match.group(2)
            else:
                # Continuation line - append to current
                if current_text is not None:
                    current_text += " " + raw_line

        # Yield final line
        if current_citation is not None and current_text is not None:
            yield TesseraeLine(
                citation=f"<{current_citation}>",
                text=current_text.strip(),
                line_number=line_num,
            )

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a Tesserae file into text chunks with metadata.

        For Tesserae files, we yield the entire file as one text chunk
        but store line information for later span creation.

        Args:
            path: Path to .tess file.

        Yields:
            Single (text, metadata) tuple per file.
        """
        raw_text = path.read_text(encoding=self._encoding)
        lines = list(self._parse_lines(raw_text))

        if not lines:
            return

        # Combine all line texts into one document
        combined_text = " ".join(line.text for line in lines)

        # Store line info in metadata for span creation
        metadata = {
            "filename": path.name,
            "path": str(path),
            "_lines": lines,  # Private: used for span creation
        }

        yield combined_text, metadata

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Docs with citation spans.

        Each Doc has a "lines" span group containing citation-annotated spans.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects with doc.spans["lines"] populated.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot create Docs with annotation_level=NONE. "
                "Use texts() for raw strings."
            )

        for path in self._iter_paths(fileids):
            for text, metadata in self._parse_file(path):
                text = self._normalize_text(text)
                doc = nlp(text)
                doc._.fileid = str(path.relative_to(self._root))
                doc._.metadata = {k: v for k, v in metadata.items() if not k.startswith("_")}

                # Create citation spans
                lines_data = metadata.get("_lines", [])
                doc.spans["lines"] = self._make_line_spans(doc, lines_data)

                yield doc

    def _make_line_spans(self, doc: "Doc", lines: list[TesseraeLine]) -> list["Span"]:
        """Create citation-annotated spans from line data.

        Maps each TesseraeLine to a character span in the combined document,
        then creates spaCy Spans with citation metadata.

        Args:
            doc: The spaCy Doc containing the combined text.
            lines: List of TesseraeLine objects with citation info.

        Returns:
            List of Spans with _.citation set.
        """
        spans = []
        char_offset = 0

        for line in lines:
            start_char = char_offset
            end_char = char_offset + len(line.text)

            # Create span using character offsets
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is not None:
                span._.citation = line.citation
                spans.append(span)

            # Move offset (+1 for the space between lines)
            char_offset = end_char + 1

        return spans

    def lines(self, fileids: str | list[str] | None = None) -> Iterator["Span"]:
        """Yield individual citation lines as Spans.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Span objects with _.citation set.
        """
        for doc in self.docs(fileids):
            yield from doc.spans.get("lines", [])

    def doc_rows(self, fileids: str | list[str] | None = None) -> Iterator[dict[str, "Span"]]:
        """Yield citation -> Span mappings for each document.

        Useful for looking up text by citation.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Dict mapping citation strings to Spans.
        """
        for doc in self.docs(fileids):
            yield {span._.citation: span for span in doc.spans.get("lines", [])}

    def texts_by_line(self, fileids: str | list[str] | None = None) -> Iterator[tuple[str, str]]:
        """Yield (citation, text) pairs. Zero NLP overhead.

        This is the fastest way to iterate over Tesserae content when you
        just need citation-text pairs without linguistic annotation.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Tuples of (citation, text).
        """
        for path in self._iter_paths(fileids):
            raw_text = path.read_text(encoding=self._encoding)
            for line in self._parse_lines(raw_text):
                yield line.citation, self._normalize_text(line.text)
