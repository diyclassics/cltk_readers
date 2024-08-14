import unittest
from cltkreaders.lat import LatinTesseraeCorpusReader
import spacy


class TestLatinTesseraeCorpusReader(unittest.TestCase):

    def setUp(self):
        self.reader = LatinTesseraeCorpusReader(
            root="cltkreaders/tests/test_root", fileids=r"tesserae.tess"
        )

    def test_fileids(self):
        self.assertEqual(self.reader.fileids(), ["tesserae.tess"])

    def test_sents(self):
        sents = list(self.reader.sents())
        sent = sents[0]
        self.assertIsInstance(sent, spacy.tokens.span.Span)
        for i, sent in enumerate(sents):
            print(i, sent.text)
        self.assertTrue(len(sents) == 0)

    def test_chunks_basis_char(self):
        chunks = self.reader.chunks(basis="char", chunk_size=100)
        chunk = next(chunks)
        print(type(chunk))
        self.assertIsInstance(chunk, spacy.tokens.doc.Doc)
        self.assertTrue(100 <= len(chunk.text) <= 200)

    def test_chunks_basis_token(self):
        chunks = self.reader.chunks(basis="token", chunk_size=100)
        chunk = next(chunks)
        print(type(chunk))
        self.assertIsInstance(chunk, spacy.tokens.doc.Doc)
        self.assertTrue(100 <= len(chunk) <= 200)

    def test_chunks_basis_token_keeptail(self):
        chunks = self.reader.chunks(basis="token", chunk_size=100, keep_tail=True)
        *_, last_chunk = chunks
        self.assertTrue(100 <= len(last_chunk))
