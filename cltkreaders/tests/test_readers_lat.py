import pytest
from cltkreaders.lat import LatinLibraryCorpusReader


@pytest.fixture(scope="module")
def latinlibrarycorpusreader():
    return LatinLibraryCorpusReader(
        root="cltkreaders/tests/test_root", fileids=r"latinlibrary.txt"
    )


def test_cltk_latinlibrarycorpusreader_fileids(latinlibrarycorpusreader):
    assert latinlibrarycorpusreader.fileids() == ["latinlibrary.txt"]


def test_cltk_plaintext_corpusreader_docs(latinlibrarycorpusreader):
    assert next(latinlibrarycorpusreader.docs())[:9] == "Q. Mucius"


def test_cltk_plaintext_corpusreader_paras(latinlibrarycorpusreader):
    assert len(list(latinlibrarycorpusreader.paras())) == 5
