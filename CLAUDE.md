# CLAUDE.md - Project Context for Claude Code

## Project Overview

**latincy-readers** (currently cltk_readers) is a specialized corpus reader library for classical Latin texts, built on top of NLTK's corpus reader architecture and leveraging spaCy's LatinCy models for NLP processing.

The project is undergoing a **ground-up refactor** with these primary goals:

1. **PyPI-ready packaging** - Best practice `src/` layout, `pyproject.toml`, CI/CD
2. **Performance-conscious API** - Smart use of generators vs. lists, lazy loading
3. **Flexible NLP integration** - Optional annotations, configurable pipeline overhead
4. **Improved metadata handling** - Centralized, validated, queryable
5. **Clearer file selection** - Redesigned `fileids` API to reduce confusion
6. **Easy extensibility** - Minimal boilerplate to add new corpus readers
7. **Latin-only focus** - Greek support spun off to separate package

## Build & Test Commands

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest cltkreaders/tests/ -v

# Run a single test file
python -m pytest cltkreaders/tests/test_readers_lat.py -v

# Run with coverage
python -m pytest cltkreaders/tests/ --cov=cltkreaders --cov-report=term-missing
```

## Project Structure

```
latincy-readers/              # New repo name (currently cltk_readers/)
├── src/
│   └── latincyreaders/       # Main package (currently cltkreaders/)
│       ├── __init__.py
│       ├── core/             # Base classes and protocols
│       │   ├── __init__.py
│       │   ├── base.py       # Abstract base reader
│       │   ├── protocols.py  # Type protocols for readers
│       │   └── mixins.py     # Shared functionality mixins
│       ├── readers/          # Concrete reader implementations
│       │   ├── __init__.py
│       │   ├── tesserae.py   # Tesserae corpus reader
│       │   ├── perseus.py    # Perseus XML/TEI readers
│       │   ├── treebank.py   # AGLDT and UD readers
│       │   ├── plaintext.py  # Generic plaintext readers
│       │   └── ...
│       ├── nlp/              # NLP pipeline integration
│       │   ├── __init__.py
│       │   └── pipeline.py   # LatinCy/spaCy integration
│       └── utils/            # Utilities
│           ├── __init__.py
│           ├── citations.py  # Citation handling
│           └── metadata.py   # Metadata management
├── tests/                    # Test suite
│   ├── conftest.py           # Pytest fixtures
│   ├── test_core/
│   ├── test_readers/
│   └── fixtures/             # Test data
├── pyproject.toml            # Modern packaging
├── CLAUDE.md                 # This file
└── DEV_PLAN.md               # Development roadmap
```

## Code Style Guidelines

### Python Style
- Python 3.10+ (use modern typing syntax: `list[str]` not `List[str]`)
- Type hints on all public methods
- Docstrings following Google style
- Line length: 88 characters (Black default)
- Use `ruff` for linting and formatting

### OO Design Principles
- Prefer composition over inheritance where possible
- Use Protocol classes for duck typing instead of abstract base classes when appropriate
- Keep inheritance hierarchies shallow (max 2-3 levels)
- Mixins should be single-responsibility

### Reader Interface Contract
All readers should implement these core methods:
```python
def fileids(self, match: str | None = None, **kwargs) -> list[str]: ...
def docs(self, fileids: str | list[str] | None = None) -> Iterator[Doc]: ...
def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]: ...
def sents(self, fileids: str | list[str] | None = None) -> Iterator[Span]: ...
def tokens(self, fileids: str | list[str] | None = None) -> Iterator[Token]: ...
```

### NLP Integration
- Default to LatinCy `la_core_web_lg` model
- All NLP-processed output should return spaCy objects (Doc, Span, Token)
- Citation information stored in spaCy custom extensions
- **Lazy loading** of NLP models (don't load until first use)
- **Configurable annotation levels**: NONE → TOKENIZE → BASIC → FULL

### Performance Guidelines
- **Generators by default** for all iteration methods (`docs()`, `sents()`, `tokens()`)
- Use `list()` explicitly when materialization needed
- Provide `texts()` for raw strings with zero NLP overhead
- Add `as_text=True` parameter to return strings instead of spaCy objects
- Document memory implications in docstrings

## Key Architectural Decisions

### 1. spaCy-First Design
The refactored library centers on spaCy/LatinCy as the primary NLP backend:
- All tokenization, lemmatization, POS tagging via spaCy
- Rich annotation preserved in spaCy objects
- Citation tracking via custom Doc/Span/Token extensions

### 2. Protocol-Based Interfaces
Use `typing.Protocol` for defining reader interfaces:
```python
class CorpusReader(Protocol):
    def fileids(self) -> list[str]: ...
    def docs(self, fileids=None) -> Iterator[Doc]: ...
```

### 3. Generator-Based Streaming
All corpus access methods return generators for memory efficiency:
```python
def sents(self, fileids=None) -> Iterator[Span]:
    for doc in self.docs(fileids):
        yield from doc.sents
```

### 4. Metadata as First-Class Citizen
- JSON-based metadata stored alongside corpora
- Faceted search via `fileids(**kwargs)` (being redesigned)
- Metadata attached to spaCy objects via extensions

### 5. Redesigned File Selection
Replace confusing `fileids()` with fluent `select()` API:
```python
# Old (confusing mixing of patterns and metadata)
reader.fileids(match="cicero", author="Cicero", min_date=-100)

# New (clear separation of concerns)
files = reader.select().match("*.tess").where(author="Vergil").between(-50, 50)
print(f"Found {files.count()} files")
```

## Dependencies

### Core
- `spacy>=3.7` - NLP framework
- `la-core-web-lg>=3.7` - LatinCy model (via HuggingFace)
- `lxml` - XML parsing
- `textacy` - spaCy text extraction utilities

### Optional/Development
- `pytest` - Testing
- `pytest-cov` - Coverage
- `ruff` - Linting/formatting
- `mypy` - Type checking

## Known Issues to Address in Refactor

### Architecture
1. **Deep Inheritance**: Some readers have 3-4 levels of inheritance → flatten to max 2
2. **Code Duplication**: `fileids()` logic duplicated across mixins → centralize in `FileSelector`
3. **Inconsistent Interfaces**: Mixed return types → always spaCy objects, `as_text` for strings

### Performance
4. **No lazy loading**: NLP model loads immediately → defer until first use
5. **Always full annotation**: No way to skip heavy NLP → add `AnnotationLevel` enum
6. **List/generator confusion**: Inconsistent materialization → generators default, explicit `list()`

### Usability
7. **`fileids` confusion**: Mixes patterns and metadata → redesign as `select()` fluent API
8. **Hard to extend**: Too much boilerplate for new readers → clear `_parse_file()` contract

### Quality
9. **Test Coverage**: Only 4 tests, with bugs → comprehensive pytest suite
10. **Error Handling**: Bare `except:` clauses → specific exceptions with useful messages
11. **Type Hints**: Incomplete → full typing with Python 3.10+ syntax

## Migration Notes

### From `cltk_readers` to `latincy-readers`
- Package rename: `cltkreaders` -> `latincyreaders`
- Greek readers removed (may become separate package)
- All readers standardized to spaCy-first interface
- Modern `pyproject.toml` packaging

### Breaking Changes Planned
- Removal of CLTK-based NLP fallbacks (spaCy only)
- Consistent return types (always spaCy objects, use `.text` for strings)
- Simplified constructor signatures
- Removal of deprecated `preprocess` callbacks (use spaCy pipelines instead)
