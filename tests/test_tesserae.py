"""Tests for TesseraeReader."""

import pytest
from pathlib import Path

from latincyreaders import TesseraeReader, AnnotationLevel


class TestTesseraeReader:
    """Test suite for TesseraeReader."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        """Create a TesseraeReader with test fixtures."""
        return TesseraeReader(root=tesserae_dir, fileids="*.tess")

    @pytest.fixture
    def reader_tokenize_only(self, tesserae_dir):
        """Reader with minimal annotation for faster tests."""
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .tess files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".tess") for f in fileids)

    def test_fileids_contains_test_file(self, reader):
        """Test fixture file is discovered."""
        fileids = reader.fileids()
        assert "tesserae.tess" in fileids

    def test_root_is_path(self, reader, tesserae_dir):
        """root property returns correct Path."""
        assert reader.root == tesserae_dir.resolve()

    # -------------------------------------------------------------------------
    # Raw text access (no NLP)
    # -------------------------------------------------------------------------

    def test_texts_yields_strings(self, reader):
        """texts() yields raw strings."""
        texts = list(reader.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contains_latin(self, reader):
        """Text content is Latin."""
        text = next(reader.texts())
        # Check for known content from test file
        assert "Mucius" in text

    def test_texts_by_line_yields_tuples(self, reader):
        """texts_by_line() yields (citation, text) tuples."""
        lines = list(reader.texts_by_line())
        assert len(lines) > 0
        assert all(isinstance(line, tuple) and len(line) == 2 for line in lines)

    def test_texts_by_line_has_citations(self, reader):
        """Citations are in correct format."""
        citation, text = next(reader.texts_by_line())
        assert citation.startswith("<")
        assert citation.endswith(">")
        assert len(text) > 0

    # -------------------------------------------------------------------------
    # spaCy Doc access
    # -------------------------------------------------------------------------

    def test_docs_yields_spacy_docs(self, reader_tokenize_only):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader_tokenize_only.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader_tokenize_only):
        """Docs have fileid extension set."""
        doc = next(reader_tokenize_only.docs())
        assert doc._.fileid is not None
        assert doc._.fileid.endswith(".tess")

    def test_docs_have_metadata(self, reader_tokenize_only):
        """Docs have metadata extension set."""
        doc = next(reader_tokenize_only.docs())
        assert doc._.metadata is not None
        assert "filename" in doc._.metadata

    def test_docs_have_line_spans(self, reader_tokenize_only):
        """Docs have 'lines' span group."""
        doc = next(reader_tokenize_only.docs())
        assert "lines" in doc.spans
        assert len(doc.spans["lines"]) > 0

    def test_line_spans_have_citations(self, reader_tokenize_only):
        """Line spans have citation extensions."""
        doc = next(reader_tokenize_only.docs())
        for span in doc.spans["lines"]:
            assert span._.citation is not None
            assert span._.citation.startswith("<")

    # -------------------------------------------------------------------------
    # Sentence and token iteration
    # -------------------------------------------------------------------------

    def test_sents_yields_spans(self, reader_tokenize_only):
        """sents() yields Span objects."""
        from spacy.tokens import Span

        sents = list(reader_tokenize_only.sents())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_sents_as_text_yields_strings(self, reader_tokenize_only):
        """sents(as_text=True) yields strings."""
        sents = list(reader_tokenize_only.sents(as_text=True))
        assert len(sents) > 0
        assert all(isinstance(s, str) for s in sents)

    def test_tokens_yields_tokens(self, reader_tokenize_only):
        """tokens() yields Token objects."""
        from spacy.tokens import Token

        tokens = list(reader_tokenize_only.tokens())
        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    def test_tokens_as_text_yields_strings(self, reader_tokenize_only):
        """tokens(as_text=True) yields strings."""
        tokens = list(reader_tokenize_only.tokens(as_text=True))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    # -------------------------------------------------------------------------
    # Citation-specific methods
    # -------------------------------------------------------------------------

    def test_lines_yields_spans_with_citations(self, reader_tokenize_only):
        """lines() yields Spans with citations."""
        from spacy.tokens import Span

        lines = list(reader_tokenize_only.lines())
        assert len(lines) > 0
        assert all(isinstance(line, Span) for line in lines)
        assert all(line._.citation is not None for line in lines)

    def test_doc_rows_yields_dicts(self, reader_tokenize_only):
        """doc_rows() yields citation->Span dicts."""
        rows = list(reader_tokenize_only.doc_rows())
        assert len(rows) > 0
        assert all(isinstance(r, dict) for r in rows)

        # Check structure
        row = rows[0]
        for citation, span in row.items():
            assert citation.startswith("<")
            assert hasattr(span, "text")

    # -------------------------------------------------------------------------
    # Annotation levels
    # -------------------------------------------------------------------------

    def test_annotation_level_none_blocks_docs(self, tesserae_dir):
        """annotation_level=NONE prevents docs() from working."""
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, tesserae_dir):
        """annotation_level=NONE still allows texts()."""
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) > 0

    def test_annotation_level_none_allows_texts_by_line(self, tesserae_dir):
        """annotation_level=NONE still allows texts_by_line()."""
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        lines = list(reader.texts_by_line())
        assert len(lines) > 0


class TestTesseraeLineParsing:
    """Test the line parsing logic specifically."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        return TesseraeReader(root=tesserae_dir)

    def test_parse_lines_extracts_citations(self, reader, sample_tess_file):
        """_parse_lines extracts citation-text pairs."""
        text = sample_tess_file.read_text()
        lines = list(reader._parse_lines(text))

        assert len(lines) == 5  # Test file has 5 citation lines
        assert lines[0].citation == "<cic. amicit. 1>"
        assert lines[1].citation == "<cic. amicit. 2>"

    def test_parse_lines_preserves_text(self, reader, sample_tess_file):
        """_parse_lines preserves the actual text content."""
        text = sample_tess_file.read_text()
        lines = list(reader._parse_lines(text))

        # First line should start with "Q. Mucius"
        assert lines[0].text.startswith("Q. Mucius augur")

    def test_parse_lines_handles_empty_input(self, reader):
        """_parse_lines handles empty input gracefully."""
        lines = list(reader._parse_lines(""))
        assert lines == []

    def test_parse_lines_handles_no_citations(self, reader):
        """_parse_lines handles text without citations."""
        lines = list(reader._parse_lines("Just some plain text\nwith no citations"))
        assert lines == []
