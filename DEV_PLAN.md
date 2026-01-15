# Development Plan: latincy-readers Refactor

## Executive Summary

This document outlines a comprehensive refactoring plan for transforming `cltk_readers` into `latincy-readers` - a modern, well-architected corpus reader library focused exclusively on Latin texts with first-class spaCy/LatinCy integration.

---

## Primary Goals

### 1. PyPI-Ready Package Structure
- Use `src/` layout (PEP 517/518 compliant)
- Modern `pyproject.toml` with proper metadata
- Versioning strategy (semantic versioning)
- CI/CD ready (GitHub Actions for testing + PyPI publish)

### 2. Performance: Lists vs. Generators
- **Default to generators** for all corpus iteration methods
- Provide explicit `.as_list()` or `list()` when materialization needed
- Document memory/performance tradeoffs clearly
- Add `batch_size` parameter for chunked processing

### 3. Lazy/Optional NLP Annotations
- **Three-tier API** for controlling NLP overhead:
  - `texts()` → raw strings, zero NLP overhead
  - `docs(annotate=False)` → spaCy Doc with tokenization only
  - `docs(annotate=True)` → full pipeline (lemma, POS, NER, etc.)
- Configurable pipeline components at reader level
- Deferred model loading (don't load until first use)

### 4. Improved Metadata Handling
- Centralized `MetadataManager` class
- Schema validation for metadata files
- Richer query API (date ranges, multiple values, negation)
- Metadata inheritance (corpus → subcorpus → file)

### 5. Clearer File Selection (`fileids` redesign)
- Rename to more intuitive API: `select()` or `filter()`
- Separate concerns: pattern matching vs. metadata filtering
- Return `FileSelection` object with preview/count methods
- Better error messages for empty selections

### 6. Extensibility: Fast Reader Development
- **Reader template/base class** that handles boilerplate
- Clear extension points (parsing, normalization, citation extraction)
- Cookbook with copy-paste examples
- Minimal required methods to implement

---

## Design Decisions

### Lazy NLP Architecture

```python
class AnnotationLevel(Enum):
    """Control how much NLP processing to apply."""
    NONE = "none"           # Raw text only
    TOKENIZE = "tokenize"   # Tokenization + sentence splitting
    BASIC = "basic"         # + lemma, POS
    FULL = "full"           # + NER, dependency parsing

class BaseCorpusReader:
    def __init__(self, ..., annotation_level: AnnotationLevel = AnnotationLevel.BASIC):
        self._annotation_level = annotation_level
        self._nlp = None  # Lazy loaded

    @property
    def nlp(self) -> Language:
        """Lazy-load NLP model on first access."""
        if self._nlp is None:
            self._nlp = self._create_pipeline()
        return self._nlp

    def _create_pipeline(self) -> Language:
        """Create spaCy pipeline based on annotation level."""
        if self._annotation_level == AnnotationLevel.NONE:
            return None

        nlp = spacy.blank("la")

        if self._annotation_level >= AnnotationLevel.TOKENIZE:
            # Add sentencizer only
            nlp.add_pipe("sentencizer")

        if self._annotation_level >= AnnotationLevel.BASIC:
            # Load full model but disable heavy components
            nlp = spacy.load("la_core_web_lg",
                           disable=["ner", "parser"] if self._annotation_level == AnnotationLevel.BASIC else [])

        return nlp

    # Simple API for common cases
    def texts(self, fileids=None) -> Iterator[str]:
        """Raw text, no NLP overhead."""
        for path in self._iter_paths(fileids):
            yield path.read_text(encoding=self._encoding)

    def sents(self, fileids=None, as_text: bool = False) -> Iterator[Span | str]:
        """Sentences - returns Spans by default, strings if as_text=True."""
        for doc in self.docs(fileids):
            for sent in doc.sents:
                yield sent.text if as_text else sent
```

### File Selection Redesign

```python
# Current (confusing)
reader.fileids(match="cicero", author="Cicero", min_date=-100)

# Proposed (clearer)
class FileSelector:
    """Fluent API for file selection."""

    def __init__(self, reader: BaseCorpusReader):
        self._reader = reader
        self._filters = []

    def match(self, pattern: str) -> "FileSelector":
        """Filter by filename pattern (glob or regex)."""
        self._filters.append(("match", pattern))
        return self

    def where(self, **kwargs) -> "FileSelector":
        """Filter by metadata fields."""
        self._filters.append(("metadata", kwargs))
        return self

    def between(self, start: int, end: int) -> "FileSelector":
        """Filter by date range."""
        self._filters.append(("date_range", (start, end)))
        return self

    def preview(self, n: int = 10) -> list[str]:
        """Preview first n matching files."""
        return self._apply()[:n]

    def count(self) -> int:
        """Count matching files without loading."""
        return len(self._apply())

    def __iter__(self) -> Iterator[str]:
        """Iterate over matching fileids."""
        yield from self._apply()

# Usage:
files = (reader.select()
         .match("*.tess")
         .where(author="Vergil")
         .between(-50, 50))

print(f"Found {files.count()} files")
for doc in reader.docs(files):
    ...
```

### Reader Extensibility Pattern

```python
class BaseCorpusReader(ABC):
    """
    To create a new reader, subclass and implement:

    Required:
        - _parse_file(path) -> yields (text, metadata) tuples

    Optional overrides:
        - _normalize_text(text) -> cleaned text
        - _extract_citations(text) -> list of Citation objects
        - _default_file_pattern() -> glob pattern for files
    """

    @abstractmethod
    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a single file. Yield (text_chunk, metadata) pairs."""
        ...

    def _normalize_text(self, text: str) -> str:
        """Override for corpus-specific text normalization."""
        return unicodedata.normalize("NFC", text)

    def _extract_citations(self, text: str) -> list[Citation]:
        """Override to extract citations from text."""
        return []

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default glob pattern for this corpus type."""
        return "*.*"


# Example: Creating a new reader is minimal
class MyCustomCorpusReader(BaseCorpusReader):
    """Reader for my custom corpus format."""

    @classmethod
    def _default_file_pattern(cls) -> str:
        return "*.txt"

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        text = path.read_text()
        # My format: first line is title, rest is content
        lines = text.split("\n", 1)
        title = lines[0]
        content = lines[1] if len(lines) > 1 else ""
        yield content, {"title": title, "filename": path.name}
```

---

## Phase 1: Foundation & Project Setup

### 1.1 Project Structure Migration
- [ ] Create new `src/latincyreaders/` package structure
- [ ] Set up `pyproject.toml` with modern packaging (PEP 517/518)
- [ ] Configure `ruff` for linting and formatting
- [ ] Configure `mypy` for type checking
- [ ] Set up `pytest` with proper test discovery

### 1.2 Core Abstractions Design

#### Protocol Definitions (`core/protocols.py`)
```python
from typing import Protocol, Iterator, runtime_checkable
from spacy.tokens import Doc, Span, Token

@runtime_checkable
class CorpusReaderProtocol(Protocol):
    """Interface contract for all corpus readers."""

    @property
    def root(self) -> str: ...

    def fileids(self, match: str | None = None, **kwargs) -> list[str]: ...
    def docs(self, fileids: str | list[str] | None = None) -> Iterator[Doc]: ...
    def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]: ...
    def sents(self, fileids: str | list[str] | None = None) -> Iterator[Span]: ...
    def tokens(self, fileids: str | list[str] | None = None) -> Iterator[Token]: ...


@runtime_checkable
class AnnotatedReaderProtocol(CorpusReaderProtocol, Protocol):
    """Extended interface for readers with linguistic annotations."""

    def tokenized_sents(self, fileids=None) -> Iterator[list[tuple[str, str, str]]]: ...
    def pos_sents(self, fileids=None) -> Iterator[list[str]]: ...
    def lemmatized_sents(self, fileids=None) -> Iterator[list[str]]: ...


@runtime_checkable
class CitationReaderProtocol(AnnotatedReaderProtocol, Protocol):
    """Interface for readers that preserve citation information."""

    def lines(self, fileids=None) -> Iterator[Span]: ...
    def doc_rows(self, fileids=None) -> Iterator[dict[str, Span]]: ...
    def concordance(self, fileids=None, basis: str = "lemma") -> dict[str, list]: ...
```

#### Base Reader Class (`core/base.py`)
```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
from spacy.tokens import Doc, Span, Token
from spacy import Language

class BaseCorpusReader(ABC):
    """Abstract base class for all Latin corpus readers."""

    def __init__(
        self,
        root: str | Path,
        fileids: str = "*",
        encoding: str = "utf-8",
        nlp: Language | str | None = None,
    ):
        self._root = Path(root)
        self._fileids_pattern = fileids
        self._encoding = encoding
        self._nlp = self._setup_nlp(nlp)
        self._metadata = self._load_metadata()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def nlp(self) -> Language:
        return self._nlp

    @abstractmethod
    def docs(self, fileids: str | list[str] | None = None) -> Iterator[Doc]:
        """Yield spaCy Doc objects for each document."""
        ...

    def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]:
        """Yield raw text for each document."""
        for doc in self.docs(fileids):
            yield doc.text

    def sents(self, fileids: str | list[str] | None = None) -> Iterator[Span]:
        """Yield sentence spans from documents."""
        for doc in self.docs(fileids):
            yield from doc.sents

    def tokens(self, fileids: str | list[str] | None = None) -> Iterator[Token]:
        """Yield individual tokens from documents."""
        for doc in self.docs(fileids):
            yield from doc
```

### 1.3 NLP Pipeline Module (`nlp/pipeline.py`)
```python
import spacy
from spacy.tokens import Doc, Span, Token
from functools import lru_cache

# Register custom extensions once at module load
if not Doc.has_extension("metadata"):
    Doc.set_extension("metadata", default=None)
    Doc.set_extension("fileid", default=None)
    Span.set_extension("citation", default=None)
    Span.set_extension("metadata", default=None)
    Token.set_extension("citation", default=None)

@lru_cache(maxsize=4)
def load_model(model_name: str = "la_core_web_lg") -> spacy.Language:
    """Load and cache a spaCy model."""
    nlp = spacy.load(model_name)
    nlp.max_length = 2_500_000  # Handle large documents
    return nlp

def get_default_nlp() -> spacy.Language:
    """Get the default LatinCy model."""
    return load_model("la_core_web_lg")
```

---

## Phase 2: Reader Implementations

### 2.1 Reader Class Hierarchy

```
BaseCorpusReader (ABC)
├── PlaintextReader
│   └── LatinLibraryReader
├── TesseraeReader (implements CitationReaderProtocol)
├── TEIReader (ABC)
│   ├── PerseusReader
│   ├── CamenaReader
│   └── CSELReader
└── TreebankReader
    ├── AGLDTReader
    └── UDReader
```

### 2.2 Tesserae Reader Refactor (`readers/tesserae.py`)

**Current Issues:**
- Deep inheritance: `LatinTesseraeCorpusReader -> CLTKLatinCorpusReaderMixin + TesseraeCorpusReader -> CLTKPlaintextCorpusReader -> PlaintextCorpusReader`
- Duplicated citation handling logic
- Mixed responsibilities

**Refactored Design:**
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import re

from spacy.tokens import Doc, Span
from ..core.base import BaseCorpusReader
from ..nlp.pipeline import get_default_nlp


@dataclass
class TesseraeLine:
    """A single citation-text pair from a Tesserae file."""
    citation: str
    text: str


class TesseraeReader(BaseCorpusReader):
    """Reader for Tesserae-format Latin texts with citation preservation."""

    CITATION_PATTERN = re.compile(r"<([^>]+)>\s*(.+)")

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str = "*.tess",
        auto_download: bool = True,
        **kwargs
    ):
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(root, fileids, **kwargs)

    def _parse_lines(self, text: str) -> Iterator[TesseraeLine]:
        """Parse citation-text pairs from Tesserae format."""
        # Handle line continuations
        text = re.sub(r"\n([^<])", r" \1", text)

        for line in text.strip().split("\n"):
            if match := self.CITATION_PATTERN.match(line):
                yield TesseraeLine(
                    citation=f"<{match.group(1)}>",
                    text=match.group(2).strip()
                )

    def docs(self, fileids: str | list[str] | None = None) -> Iterator[Doc]:
        """Yield annotated spaCy Docs with citation spans."""
        for path in self._iter_paths(fileids):
            raw_text = path.read_text(encoding=self._encoding)
            lines = list(self._parse_lines(raw_text))

            # Combine texts and track offsets
            combined_text = " ".join(line.text for line in lines)
            doc = self.nlp(combined_text)

            # Add citation spans
            doc.spans["lines"] = self._make_line_spans(doc, lines)
            doc._.fileid = path.name
            doc._.metadata = self._metadata.get(path.name, {})

            yield doc

    def lines(self, fileids=None) -> Iterator[Span]:
        """Yield individual lines as Spans with citations."""
        for doc in self.docs(fileids):
            yield from doc.spans.get("lines", [])

    def doc_rows(self, fileids=None) -> Iterator[dict[str, Span]]:
        """Yield citation -> Span mappings for each document."""
        for doc in self.docs(fileids):
            yield {
                span._.citation: span
                for span in doc.spans.get("lines", [])
            }
```

### 2.3 Perseus TEI Reader Refactor (`readers/perseus.py`)

**Current Issues:**
- XML namespace handling scattered across methods
- Note removal logic duplicated
- Inconsistent paragraph extraction

**Refactored Design:**
```python
from lxml import etree
from pathlib import Path
from typing import Iterator
from spacy.tokens import Doc

from ..core.base import BaseCorpusReader


class TEIReader(BaseCorpusReader):
    """Base reader for TEI/XML documents."""

    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

    def __init__(
        self,
        root: str | Path,
        fileids: str = "*.xml",
        namespaces: dict | None = None,
        remove_notes: bool = True,
        **kwargs
    ):
        self._ns = namespaces or {}
        self._remove_notes = remove_notes
        super().__init__(root, fileids, **kwargs)

    def _parse_xml(self, path: Path) -> etree._Element:
        """Parse XML with huge_tree support."""
        parser = etree.XMLParser(huge_tree=True, encoding="utf-8")
        return etree.parse(str(path), parser).getroot()

    def _get_body(self, root: etree._Element) -> etree._Element:
        """Extract TEI body, removing notes if configured."""
        body = root.find(".//body", self._ns) or root.find(".//tei:body", self.TEI_NS)

        if self._remove_notes and body is not None:
            for note in body.iterfind(".//note", self._ns):
                note.getparent().remove(note)
            for note in body.iterfind(".//tei:note", self.TEI_NS):
                note.getparent().remove(note)

        return body

    def _extract_paragraphs(self, body: etree._Element) -> list[str]:
        """Extract paragraph texts from body element."""
        paras = body.findall(".//p", self._ns) or body.findall(".//tei:p", self.TEI_NS)
        if not paras:
            paras = [body]
        return [" ".join(p.itertext()) for p in paras]


class PerseusReader(TEIReader):
    """Reader for Perseus Digital Library XML files."""

    def docs(self, fileids: str | list[str] | None = None) -> Iterator[Doc]:
        for path in self._iter_paths(fileids):
            root = self._parse_xml(path)
            body = self._get_body(root)

            if body is None:
                continue

            paragraphs = self._extract_paragraphs(body)
            full_text = "\n\n".join(paragraphs)

            doc = self.nlp(full_text)
            doc._.fileid = path.name
            doc._.metadata = self._metadata.get(path.name, {})

            yield doc
```

### 2.4 Treebank Readers (`readers/treebank.py`)

**AGLDT Reader:**
```python
class AGLDTReader(TEIReader):
    """Reader for Perseus AGLDT (Ancient Greek and Latin Dependency Treebank) files."""

    def docs(self, fileids=None) -> Iterator[Doc]:
        for path in self._iter_paths(fileids):
            root = self._parse_xml(path)

            for sentence_elem in root.iterfind(".//sentence"):
                words = sentence_elem.findall(".//word")
                tokens = [w.get("form", "") for w in words]
                lemmas = [w.get("lemma", "") for w in words]
                postags = [w.get("postag", "") for w in words]

                # Create Doc with pre-tokenized words
                doc = Doc(self.nlp.vocab, words=tokens)
                # ... set annotations
                yield doc
```

**Universal Dependencies Reader:**
```python
class UDReader(BaseCorpusReader):
    """Reader for CoNLL-U format Universal Dependencies files."""

    COLUMNS = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]

    def _parse_conllu(self, text: str) -> Iterator[list[dict]]:
        """Parse CoNLL-U format into sentence dictionaries."""
        for block in text.strip().split("\n\n"):
            sentence = []
            for line in block.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.split("\t")
                sentence.append(dict(zip(self.COLUMNS, fields)))
            if sentence:
                yield sentence
```

---

## Phase 3: Shared Functionality

### 3.1 Metadata Management (`utils/metadata.py`)
```python
from pathlib import Path
import json
from typing import Any
from functools import cached_property


class MetadataManager:
    """Manages corpus metadata from JSON files."""

    def __init__(self, root: Path):
        self.root = root

    @cached_property
    def metadata(self) -> dict[str, dict[str, Any]]:
        """Load and merge all metadata files."""
        merged = {}
        for json_file in self.root.rglob("metadata/*.json"):
            data = json.loads(json_file.read_text())
            for key, value in data.items():
                merged.setdefault(key, {}).update(value)
        return merged

    def get(self, fileid: str, default: dict | None = None) -> dict:
        """Get metadata for a specific file."""
        return self.metadata.get(fileid, default or {})

    def filter_fileids(
        self,
        fileids: list[str],
        *,
        match: str | None = None,
        min_date: int | None = None,
        max_date: int | None = None,
        **kwargs
    ) -> list[str]:
        """Filter fileids by metadata criteria."""
        result = set(fileids)

        if match:
            import re
            result &= {f for f in result if re.search(match, f, re.IGNORECASE)}

        for key, value in kwargs.items():
            result &= {
                f for f in result
                if self._matches(self.get(f), key, str(value))
            }

        if min_date is not None or max_date is not None:
            result &= self._filter_by_date(result, min_date, max_date)

        return sorted(result)
```

### 3.2 Citation Utilities (`utils/citations.py`)
```python
from dataclasses import dataclass
from typing import Iterator
from spacy.tokens import Doc, Span


@dataclass(frozen=True)
class Citation:
    """Represents a textual citation."""
    text: str
    start_char: int
    end_char: int

    def __str__(self) -> str:
        return self.text


def make_citation_spans(
    doc: Doc,
    citations: list[str],
    char_offsets: list[tuple[int, int]]
) -> list[Span]:
    """Create citation-annotated Spans from character offsets."""
    spans = []
    for citation, (start, end) in zip(citations, char_offsets):
        span = doc.char_span(start, end, alignment_mode="expand")
        if span:
            span._.citation = citation
            spans.append(span)
    return spans


def build_concordance(
    tokens: Iterator,
    basis: str = "lemma",
    only_alpha: bool = True
) -> dict[str, list]:
    """Build a concordance from token stream."""
    from collections import defaultdict
    concordance = defaultdict(list)

    for token in tokens:
        if only_alpha and not token.is_alpha:
            continue

        key = getattr(token, f"{basis}_", token.text)
        concordance[key].append(token._.citation)

    return dict(sorted(concordance.items()))
```

---

## Phase 4: Testing Strategy

### 4.1 Test Structure
```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/                # Test data
│   ├── tesserae/
│   │   └── sample.tess
│   ├── perseus/
│   │   └── sample.xml
│   └── ud/
│       └── sample.conllu
├── test_core/
│   ├── test_base.py
│   └── test_protocols.py
├── test_readers/
│   ├── test_tesserae.py
│   ├── test_perseus.py
│   ├── test_treebank.py
│   └── test_plaintext.py
├── test_utils/
│   ├── test_metadata.py
│   └── test_citations.py
└── test_integration/
    └── test_full_pipeline.py
```

### 4.2 Test Fixtures (`conftest.py`)
```python
import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_tesserae(fixtures_dir) -> Path:
    return fixtures_dir / "tesserae" / "sample.tess"

@pytest.fixture
def tesserae_reader(sample_tesserae):
    from latincyreaders.readers.tesserae import TesseraeReader
    return TesseraeReader(root=sample_tesserae.parent)

@pytest.fixture
def nlp():
    """Provide shared NLP model for tests."""
    from latincyreaders.nlp.pipeline import get_default_nlp
    return get_default_nlp()
```

### 4.3 Example Tests
```python
# test_readers/test_tesserae.py

import pytest
from spacy.tokens import Doc, Span

class TestTesseraeReader:

    def test_fileids_returns_list(self, tesserae_reader):
        fileids = tesserae_reader.fileids()
        assert isinstance(fileids, list)
        assert all(f.endswith(".tess") for f in fileids)

    def test_docs_yields_spacy_docs(self, tesserae_reader):
        docs = list(tesserae_reader.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_citation_spans(self, tesserae_reader):
        doc = next(tesserae_reader.docs())
        assert "lines" in doc.spans
        assert all(span._.citation for span in doc.spans["lines"])

    def test_sents_yields_spans(self, tesserae_reader):
        sents = list(tesserae_reader.sents())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_tokens_have_citations(self, tesserae_reader):
        tokens = list(tesserae_reader.tokens())[:10]
        # At least some tokens should have citation info
        assert any(t._.citation for t in tokens)

    def test_concordance_structure(self, tesserae_reader):
        conc = tesserae_reader.concordance(basis="lemma")
        assert isinstance(conc, dict)
        # Values should be lists of citation tuples
        for key, refs in conc.items():
            assert isinstance(key, str)
            assert isinstance(refs, list)

    def test_chunks_respects_size(self, tesserae_reader):
        chunks = list(tesserae_reader.chunks(chunk_size=100, basis="token"))
        for chunk in chunks[:-1]:  # All but last
            assert len(chunk) >= 100
```

---

## Phase 5: Documentation & Packaging

### 5.1 pyproject.toml
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "latincy-readers"
version = "1.0.0"
description = "Corpus readers for Latin texts with LatinCy/spaCy integration"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Patrick J. Burns", email = "patrick@diyclassics.org" }
]
keywords = ["latin", "nlp", "corpus", "spacy", "classical-languages"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Text Processing :: Linguistic",
]

dependencies = [
    "spacy>=3.7",
    "lxml>=5.0",
    "natsort>=8.0",
    "pyuca>=1.2",
    "textacy>=0.13",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
models = [
    "la-core-web-lg @ https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl",
]

[project.urls]
Homepage = "https://github.com/diyclassics/latincy-readers"
Documentation = "https://latincy-readers.readthedocs.io"
Repository = "https://github.com/diyclassics/latincy-readers"

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## Implementation Roadmap

### Milestone 1: Core Infrastructure
- [ ] Set up new project structure with `src/` layout
- [ ] Implement `pyproject.toml` packaging
- [ ] Create `core/protocols.py` with interface definitions
- [ ] Create `core/base.py` with `BaseCorpusReader`
- [ ] Implement `nlp/pipeline.py` with model loading
- [ ] Set up pytest with basic fixtures

### Milestone 2: Primary Readers
- [ ] Implement `TesseraeReader` (highest priority)
- [ ] Implement `PerseusReader` (TEI/XML base)
- [ ] Implement `PlaintextReader` and `LatinLibraryReader`
- [ ] Achieve 80%+ test coverage on new code

### Milestone 3: Specialized Readers
- [ ] Implement `AGLDTReader` for treebanks
- [ ] Implement `UDReader` for Universal Dependencies
- [ ] Implement `CamenaReader` and `CSELReader`
- [ ] Full test coverage

### Milestone 4: Polish & Release
- [ ] Complete type annotations
- [ ] Comprehensive docstrings
- [ ] README and documentation
- [ ] Migration guide from `cltk_readers`
- [ ] Release v1.0.0

---

## Notes for Development

### Things to Preserve
- Metadata-based file filtering (very useful feature)
- Citation tracking through spaCy extensions
- Generator-based memory efficiency
- KWIC and concordance functionality
- Chunking with sentence boundary respect

### Things to Remove/Deprecate
- CLTK NLP fallback (spaCy only going forward)
- Greek reader support (separate package)
- `preprocess` callback parameters (use spaCy components)
- Dual return types (always spaCy objects)

### Potential Future Enhancements
- Async corpus loading for large collections
- Built-in caching for processed documents
- Export to HuggingFace datasets format
- Integration with sentence transformers for embeddings
