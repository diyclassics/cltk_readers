"""Corpus readers to support Latin text collections for use with CLTK; see readers.py for more information
"""

__author__ = ["Patrick J. Burns <patrick@diyclassics.org>",]
__license__ = "MIT License."

from typing import Callable, Iterator, Union

from cltkreaders.readers import TesseraeCorpusReader

from cltk import NLP
from cltk.core.data_types import Pipeline
from cltk.languages.utils import get_lang
from cltk.alphabet.processes import LatinNormalizeProcess
from cltk.dependency.processes import LatinStanzaProcess
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
        
        pipeline = Pipeline(description="Latin pipeline for Tesserae readers", 
                            processes=[LatinNormalizeProcess, LatinStanzaProcess], 
                            language=get_lang(self.lang))
        self.nlp = NLP(language=lang, custom_pipeline=pipeline, suppress_banner=True)

        if not word_tokenizer:
            self.word_tokenizer = LatinWordTokenizer()
        else:
            self.word_tokenizer = word_tokenizer

        if not sent_tokenizer:
            self.sent_tokenizer = LatinPunktSentenceTokenizer()
        else:
            self.sent_tokenizer = sent_tokenizer

        
        TesseraeCorpusReader.__init__(self, root, fileids, encoding, self.lang,
                                      word_tokenizer=self.word_tokenizer,
                                      sent_tokenizer=self.sent_tokenizer)

    def pos_sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        for sent in self.sents(fileids):
            data = self.nlp.analyze(text=sent)
            pos_sent = []
            for item in data:
                pos_sent.append(f"{item.string}/{item.upos}")
            yield pos_sent       
