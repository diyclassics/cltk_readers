"""Tests for TEI and Perseus readers."""

import pytest
from pathlib import Path

from latincyreaders import TEIReader, PerseusReader, AnnotationLevel


class TestTEIReader:
    """Test suite for TEIReader."""

    @pytest.fixture
    def reader(self, tei_dir):
        """Create a TEIReader with test fixtures."""
        return TEIReader(root=tei_dir, fileids="*.xml")

    @pytest.fixture
    def reader_tokenize_only(self, tei_dir):
        """Reader with minimal annotation for faster tests."""
        return TEIReader(
            root=tei_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .xml files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".xml") for f in fileids)

    def test_fileids_contains_test_file(self, reader):
        """Test fixture file is discovered."""
        fileids = reader.fileids()
        assert "sample.xml" in fileids

    def test_root_is_path(self, reader, tei_dir):
        """root property returns correct Path."""
        assert reader.root == tei_dir.resolve()

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

    def test_texts_removes_notes_by_default(self, reader):
        """Notes are removed from extracted text by default."""
        text = next(reader.texts())
        # The note in our sample says "This is an editorial note"
        assert "editorial note" not in text

    def test_texts_preserves_notes_when_configured(self, tei_dir):
        """Notes are preserved when remove_notes=False."""
        reader = TEIReader(root=tei_dir, remove_notes=False)
        text = next(reader.texts())
        assert "editorial note" in text

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
        assert doc._.fileid.endswith(".xml")

    def test_docs_have_metadata(self, reader_tokenize_only):
        """Docs have metadata extension set."""
        doc = next(reader_tokenize_only.docs())
        assert doc._.metadata is not None
        assert "filename" in doc._.metadata

    def test_docs_have_title_in_metadata(self, reader_tokenize_only):
        """Docs extract title from TEI header."""
        doc = next(reader_tokenize_only.docs())
        metadata = doc._.metadata
        assert "title" in metadata
        assert "De Amicitia" in metadata["title"]

    def test_docs_have_author_in_metadata(self, reader_tokenize_only):
        """Docs extract author from TEI header."""
        doc = next(reader_tokenize_only.docs())
        metadata = doc._.metadata
        assert "author" in metadata
        assert "Cicero" in metadata["author"]

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
    # Paragraph iteration
    # -------------------------------------------------------------------------

    def test_paras_yields_spans(self, reader_tokenize_only):
        """paras() yields Span objects."""
        from spacy.tokens import Span

        paras = list(reader_tokenize_only.paras())
        assert len(paras) > 0
        assert all(isinstance(p, Span) for p in paras)

    def test_paras_as_text_yields_strings(self, reader_tokenize_only):
        """paras(as_text=True) yields strings."""
        paras = list(reader_tokenize_only.paras(as_text=True))
        assert len(paras) > 0
        assert all(isinstance(p, str) for p in paras)

    def test_paras_count_matches_document_structure(self, reader_tokenize_only):
        """Number of paragraphs matches expected structure."""
        # Sample.xml has 4 <p> elements
        paras = list(reader_tokenize_only.paras(as_text=True))
        assert len(paras) == 4

    # -------------------------------------------------------------------------
    # Annotation levels
    # -------------------------------------------------------------------------

    def test_annotation_level_none_blocks_docs(self, tei_dir):
        """annotation_level=NONE prevents docs() from working."""
        reader = TEIReader(
            root=tei_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, tei_dir):
        """annotation_level=NONE still allows texts()."""
        reader = TEIReader(
            root=tei_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) > 0

    def test_annotation_level_none_allows_paras_as_text(self, tei_dir):
        """annotation_level=NONE allows paras(as_text=True)."""
        reader = TEIReader(
            root=tei_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.NONE,
        )
        paras = list(reader.paras(as_text=True))
        assert len(paras) > 0


class TestTEIXMLParsing:
    """Test the XML parsing logic specifically."""

    @pytest.fixture
    def reader(self, tei_dir):
        return TEIReader(root=tei_dir)

    def test_parse_xml_returns_element(self, reader, sample_tei_file):
        """_parse_xml returns parsed XML element."""
        from lxml import etree

        root = reader._parse_xml(sample_tei_file)
        assert root is not None
        assert isinstance(root, etree._Element)

    def test_parse_xml_handles_invalid_file(self, reader, tei_dir):
        """_parse_xml returns None for invalid XML."""
        # Create a path to non-existent file
        result = reader._parse_xml(tei_dir / "nonexistent.xml")
        assert result is None

    def test_get_body_extracts_body(self, reader, sample_tei_file):
        """_get_body extracts body element."""
        root = reader._parse_xml(sample_tei_file)
        body = reader._get_body(root)
        assert body is not None

    def test_extract_paragraphs_finds_paragraphs(self, reader, sample_tei_file):
        """_extract_paragraphs finds paragraph elements."""
        root = reader._parse_xml(sample_tei_file)
        body = reader._get_body(root)
        paras = reader._extract_paragraphs(body)
        assert len(paras) == 4  # Sample has 4 paragraphs

    def test_namespace_handling(self, reader, sample_tei_file):
        """Reader handles TEI namespace correctly."""
        root = reader._parse_xml(sample_tei_file)
        # Should find header with namespace
        header = reader._find_with_ns(root, ".//teiHeader")
        assert header is not None


class TestPerseusReader:
    """Test suite for PerseusReader."""

    @pytest.fixture
    def reader(self, tei_dir):
        """Create a PerseusReader with test fixtures."""
        return PerseusReader(root=tei_dir, fileids="*.xml")

    @pytest.fixture
    def reader_tokenize_only(self, tei_dir):
        """Reader with minimal annotation for faster tests."""
        return PerseusReader(
            root=tei_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    # -------------------------------------------------------------------------
    # Basic functionality (inherits from TEIReader)
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .xml files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0

    def test_texts_yields_strings(self, reader):
        """texts() yields raw strings."""
        texts = list(reader.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_docs_yields_spacy_docs(self, reader_tokenize_only):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader_tokenize_only.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    # -------------------------------------------------------------------------
    # Perseus-specific: headers() method
    # -------------------------------------------------------------------------

    def test_headers_yields_dicts(self, reader):
        """headers() yields metadata dictionaries."""
        headers = list(reader.headers())
        assert len(headers) > 0
        assert all(isinstance(h, dict) for h in headers)

    def test_headers_contain_filename(self, reader):
        """headers() includes filename in metadata."""
        header = next(reader.headers())
        assert "filename" in header
        assert header["filename"].endswith(".xml")

    def test_headers_extract_title(self, reader):
        """headers() extracts title from TEI header."""
        header = next(reader.headers())
        assert "title" in header
        assert "De Amicitia" in header["title"]

    def test_headers_extract_author(self, reader):
        """headers() extracts author from TEI header."""
        header = next(reader.headers())
        assert "author" in header
        assert "Cicero" in header["author"]
