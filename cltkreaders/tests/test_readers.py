import pytest
from cltkreaders.readers import CLTKPlaintextCorpusReader


@pytest.fixture(scope="module")
def corpusreader():
    return CLTKPlaintextCorpusReader(
        root="cltkreaders/tests/test_root", fileids=r".*\.txt"
    )


def test_cltk_plaintext_corpusreader_fileids(corpusreader):
    assert corpusreader.fileids() == ["latinlibrary.txt"]


def test_cltk_plaintext_corpusreader_docs(corpusreader):
    assert next(corpusreader.docs())[:9] == "Q. Mucius"


def test_cltk_plaintext_corpusreader_paras(corpusreader):
    assert len(list(corpusreader.paras())) == 5
