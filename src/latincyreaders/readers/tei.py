"""TEI/XML corpus readers.

Readers for TEI (Text Encoding Initiative) XML files:
- TEIReader: Base reader for TEI/XML documents
- PerseusReader: Reader for Perseus Digital Library XML files
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from lxml import etree

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


class TEIReader(BaseCorpusReader):
    """Base reader for TEI/XML documents.

    Handles common TEI structure including namespace handling, body extraction,
    and note removal. Subclass for corpus-specific implementations.

    Example:
        >>> reader = TEIReader("/path/to/tei")
        >>> for doc in reader.docs():
        ...     print(doc.text[:100])
    """

    # Common TEI namespace
    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default to .xml files."""
        return "**/*.xml"

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.BASIC,
        namespaces: dict[str, str] | None = None,
        remove_notes: bool = True,
    ):
        """Initialize the TEI reader.

        Args:
            root: Root directory containing TEI files.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            namespaces: XML namespaces to use. If None, tries both with and without TEI namespace.
            remove_notes: Whether to remove <note> elements from text.
        """
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
        )
        self._namespaces = namespaces
        self._remove_notes = remove_notes

    def _parse_xml(self, path: Path) -> etree._Element | None:
        """Parse XML file with huge_tree support.

        Args:
            path: Path to XML file.

        Returns:
            Parsed XML root element, or None if parsing fails.
        """
        try:
            parser = etree.XMLParser(huge_tree=True, encoding="utf-8")
            tree = etree.parse(str(path), parser)
            return tree.getroot()
        except (etree.Error, OSError):
            return None

    def _add_namespace_prefix(self, xpath: str, prefix: str = "tei") -> str:
        """Add namespace prefix to XPath element names.

        Args:
            xpath: XPath expression (e.g., ".//body" or ".//teiHeader//title").
            prefix: Namespace prefix to add.

        Returns:
            XPath with namespace prefixes added to element names.
        """
        import re
        # Match element names after . // or /
        # Pattern: match element names (letters) that follow .// or // or /
        # But not if they already have a prefix (contain :)
        result = re.sub(
            r'((?:^|/|\.)/)([a-zA-Z][a-zA-Z0-9]*)',
            rf'\1{prefix}:\2',
            xpath
        )
        return result

    def _find_with_ns(self, element: etree._Element, xpath: str) -> etree._Element | None:
        """Find element trying both with and without TEI namespace.

        Args:
            element: Parent element to search in.
            xpath: XPath expression (without namespace prefix).

        Returns:
            Found element or None.
        """
        ns_xpath = self._add_namespace_prefix(xpath)

        if self._namespaces:
            result = element.find(ns_xpath, self._namespaces)
            if result is not None:
                return result

        # Try with TEI namespace
        result = element.find(ns_xpath, self.TEI_NS)
        if result is not None:
            return result

        # Try without namespace
        return element.find(xpath)

    def _findall_with_ns(self, element: etree._Element, xpath: str) -> list[etree._Element]:
        """Find all elements trying both with and without TEI namespace.

        Args:
            element: Parent element to search in.
            xpath: XPath expression (without namespace prefix).

        Returns:
            List of found elements.
        """
        ns_xpath = self._add_namespace_prefix(xpath)

        if self._namespaces:
            result = element.findall(ns_xpath, self._namespaces)
            if result:
                return result

        # Try with TEI namespace
        result = element.findall(ns_xpath, self.TEI_NS)
        if result:
            return result

        # Try without namespace
        return element.findall(xpath) or []

    def _get_body(self, root: etree._Element) -> etree._Element | None:
        """Extract TEI body, optionally removing notes.

        Args:
            root: XML root element.

        Returns:
            Body element with notes removed if configured.
        """
        body = self._find_with_ns(root, ".//body")
        if body is None:
            return None

        if self._remove_notes:
            # Remove note elements
            for notes_xpath in [".//note", ".//tei:note"]:
                try:
                    for note in body.xpath(notes_xpath, namespaces=self.TEI_NS):
                        note.getparent().remove(note)
                except Exception:
                    pass

        return body

    def _extract_paragraphs(self, body: etree._Element) -> list[str]:
        """Extract paragraph texts from body element.

        Args:
            body: TEI body element.

        Returns:
            List of paragraph text strings.
        """
        paras = self._findall_with_ns(body, ".//p")
        if not paras:
            # If no paragraphs, treat whole body as one unit
            paras = [body]

        return [" ".join(p.itertext()).strip() for p in paras if p is not None]

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a TEI file into text with metadata.

        Args:
            path: Path to XML file.

        Yields:
            Single (text, metadata) tuple per file.
        """
        root = self._parse_xml(path)
        if root is None:
            return

        body = self._get_body(root)
        if body is None:
            return

        paragraphs = self._extract_paragraphs(body)
        if not paragraphs:
            return

        # Join paragraphs with double newlines
        text = "\n\n".join(paragraphs)
        text = self._normalize_text(text)

        if not text.strip():
            return

        # Extract metadata from header
        metadata = {
            "filename": path.name,
            "path": str(path),
        }

        # Try to get title and author from teiHeader
        header = self._find_with_ns(root, ".//teiHeader")
        if header is not None:
            title_elem = self._find_with_ns(header, ".//title")
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text.strip()

            author_elem = self._find_with_ns(header, ".//author")
            if author_elem is not None and author_elem.text:
                metadata["author"] = author_elem.text.strip()

        yield text, metadata

    def paras(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield paragraphs from TEI documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Span objects.

        Yields:
            Paragraph Spans (or strings if as_text=True).
        """
        for path in self._iter_paths(fileids):
            root = self._parse_xml(path)
            if root is None:
                continue

            body = self._get_body(root)
            if body is None:
                continue

            para_texts = self._extract_paragraphs(body)

            if as_text:
                for para_text in para_texts:
                    yield self._normalize_text(para_text)
            else:
                nlp = self.nlp
                if nlp is None:
                    raise ValueError(
                        "Cannot create paragraph Spans with annotation_level=NONE. "
                        "Use paras(as_text=True) or set a higher annotation level."
                    )
                for para_text in para_texts:
                    doc = nlp(self._normalize_text(para_text))
                    yield doc[:]


class PerseusReader(TEIReader):
    """Reader for Perseus Digital Library XML files.

    Handles Perseus-specific TEI structure including PDILL collection files.

    Example:
        >>> reader = PerseusReader("/path/to/perseus")
        >>> for doc in reader.docs():
        ...     print(f"{doc._.metadata.get('author')}: {len(list(doc.sents))} sentences")
    """

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.BASIC,
        remove_notes: bool = True,
    ):
        """Initialize the Perseus reader.

        Args:
            root: Root directory containing Perseus XML files.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            remove_notes: Whether to remove <note> elements.
        """
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
            namespaces=None,  # Perseus files often don't use namespaces
            remove_notes=remove_notes,
        )

    def headers(self, fileids: str | list[str] | None = None) -> Iterator[dict]:
        """Yield metadata from TEI headers.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Dict of header metadata.
        """
        for path in self._iter_paths(fileids):
            root = self._parse_xml(path)
            if root is None:
                continue

            header = self._find_with_ns(root, ".//teiHeader")
            if header is None:
                continue

            metadata = {"filename": path.name}

            # Extract common metadata fields
            title_elem = self._find_with_ns(header, ".//title")
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text.strip()

            author_elem = self._find_with_ns(header, ".//author")
            if author_elem is not None and author_elem.text:
                metadata["author"] = author_elem.text.strip()

            yield metadata
