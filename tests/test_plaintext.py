"""Tests for PlaintextReader and LatinLibraryReader."""

import pytest
from pathlib import Path

from latincyreaders import PlaintextReader, LatinLibraryReader, AnnotationLevel


@pytest.fixture
def plaintext_dir(fixtures_dir) -> Path:
    """Path to plaintext test fixtures."""
    return fixtures_dir / "plaintext"


@pytest.fixture
def sample_txt_file(plaintext_dir) -> Path:
    """Path to sample .txt file."""
    return plaintext_dir / "latinlibrary.txt"


class TestPlaintextReader:
    """Test suite for PlaintextReader."""

    @pytest.fixture
    def reader(self, plaintext_dir):
        """Create a PlaintextReader with test fixtures."""
        return PlaintextReader(root=plaintext_dir, fileids="*.txt")

    @pytest.fixture
    def reader_tokenize(self, plaintext_dir):
        """Reader with minimal annotation."""
        return PlaintextReader(
            root=plaintext_dir,
            fileids="*.txt",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .txt files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".txt") for f in fileids)

    def test_root_is_path(self, reader, plaintext_dir):
        """root property returns correct Path."""
        assert reader.root == plaintext_dir.resolve()

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

    # -------------------------------------------------------------------------
    # spaCy Doc access
    # -------------------------------------------------------------------------

    def test_docs_yields_spacy_docs(self, reader_tokenize):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader_tokenize.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader_tokenize):
        """Docs have fileid extension set."""
        doc = next(reader_tokenize.docs())
        assert doc._.fileid is not None
        assert doc._.fileid.endswith(".txt")

    def test_docs_have_metadata(self, reader_tokenize):
        """Docs have metadata extension set."""
        doc = next(reader_tokenize.docs())
        assert doc._.metadata is not None
        assert "filename" in doc._.metadata

    # -------------------------------------------------------------------------
    # Sentence and token iteration
    # -------------------------------------------------------------------------

    def test_sents_yields_spans(self, reader_tokenize):
        """sents() yields Span objects."""
        from spacy.tokens import Span

        sents = list(reader_tokenize.sents())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_sents_as_text_yields_strings(self, reader_tokenize):
        """sents(as_text=True) yields strings."""
        sents = list(reader_tokenize.sents(as_text=True))
        assert len(sents) > 0
        assert all(isinstance(s, str) for s in sents)

    def test_tokens_yields_tokens(self, reader_tokenize):
        """tokens() yields Token objects."""
        from spacy.tokens import Token

        tokens = list(reader_tokenize.tokens())
        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    # -------------------------------------------------------------------------
    # Paragraph access
    # -------------------------------------------------------------------------

    def test_paras_as_text_yields_strings(self, reader):
        """paras(as_text=True) yields paragraph strings."""
        paras = list(reader.paras(as_text=True))
        assert len(paras) > 0
        assert all(isinstance(p, str) for p in paras)

    def test_paras_yields_spans(self, reader_tokenize):
        """paras() yields Span objects."""
        from spacy.tokens import Span

        paras = list(reader_tokenize.paras())
        assert len(paras) > 0
        assert all(isinstance(p, Span) for p in paras)

    # -------------------------------------------------------------------------
    # Annotation levels
    # -------------------------------------------------------------------------

    def test_annotation_level_none_blocks_docs(self, plaintext_dir):
        """annotation_level=NONE prevents docs() from working."""
        reader = PlaintextReader(
            root=plaintext_dir,
            fileids="*.txt",
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, plaintext_dir):
        """annotation_level=NONE still allows texts()."""
        reader = PlaintextReader(
            root=plaintext_dir,
            fileids="*.txt",
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) > 0

    def test_annotation_level_none_allows_paras_as_text(self, plaintext_dir):
        """annotation_level=NONE allows paras(as_text=True)."""
        reader = PlaintextReader(
            root=plaintext_dir,
            fileids="*.txt",
            annotation_level=AnnotationLevel.NONE,
        )
        paras = list(reader.paras(as_text=True))
        assert len(paras) > 0


class TestLatinLibraryReader:
    """Test suite for LatinLibraryReader."""

    @pytest.fixture
    def reader(self, plaintext_dir):
        """Create a LatinLibraryReader with test fixtures."""
        return LatinLibraryReader(root=plaintext_dir, fileids="*.txt")

    def test_inherits_from_plaintext(self):
        """LatinLibraryReader inherits from PlaintextReader."""
        assert issubclass(LatinLibraryReader, PlaintextReader)

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0

    def test_docs_have_title_metadata(self, reader):
        """Latin Library docs include title in metadata."""
        reader._annotation_level = AnnotationLevel.TOKENIZE
        reader._nlp = None  # Reset lazy loading

        doc = next(reader.docs())
        assert "title" in doc._.metadata

    def test_missing_root_raises_error(self):
        """Missing corpus without auto_download raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            LatinLibraryReader(root=None, auto_download=False)

    def test_texts_cleaned(self, reader):
        """Latin Library texts are cleaned."""
        text = next(reader.texts())
        # Should not have "The Latin Library" header if cleaned
        assert "the latin library" not in text.lower().split("\n")[0]
