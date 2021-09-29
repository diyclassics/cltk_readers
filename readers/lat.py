"""Corpus readers to support Latin text collections for use with CLTK; see readers.py for more information
"""

__author__ = ["Patrick J. Burns <patrick@diyclassics.org>",]
__license__ = "MIT License."

from typing import Callable

from readers import TesseraeCorpusReader

from cltk.sentence.lat import LatinPunktSentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer

class LatinTesseraeCorpusReader(TesseraeCorpusReader):
    """
    A corpus reader for Latin texts from the Tesserae-CLTK corpus
    """

    def __init__(self, root: str, fileids: str = None, encoding: str = 'utf-8', lang: str = 'lat',
                 normalization_form: str = 'NFC',
                 word_tokenizer: Callable = None, sent_tokenizer: Callable = None, **kwargs):
        """
        :param root: Location of plaintext files to be read into corpus reader
        :param fileids: Pattern for matching files to be read into corpus reader
        :param encoding: Text encoding for associated files; defaults to 'utf-8'
        :param lang: Allows a language to be selected for language-specific corpus tasks; default is 'lat'
        :param normalization_form: Normalization form for associated files; defaults to 'NFC'
        :param kwargs: Miscellaneous keyword arguments
        """
        self.lang = lang
        if not word_tokenizer:
            self.word_tokenizer = LatinWordTokenizer()
        if not sent_tokenizer:
            self.sent_tokenizer = LatinPunktSentenceTokenizer()

        TesseraeCorpusReader.__init__(self, root, fileids, encoding, self.lang,
                                      word_tokenizer=self.word_tokenizer,
                                      sent_tokenizer=self.sent_tokenizer)
