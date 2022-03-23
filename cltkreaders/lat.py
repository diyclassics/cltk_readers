"""Corpus readers to support Latin text collections for use with CLTK; see readers.py for more information
"""

__author__ = ["Patrick J. Burns <patrick@diyclassics.org>",]
__license__ = "MIT License."

import os.path
from typing import Callable, Iterator, Union
from cltkreaders.readers import TesseraeCorpusReader

from cltk import NLP
from cltk.core.data_types import Pipeline
from cltk.languages.utils import get_lang
from cltk.alphabet.processes import LatinNormalizeProcess
from cltk.dependency.processes import LatinStanzaProcess
from cltk.sentence.lat import LatinPunktSentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer

from cltk.utils import get_cltk_data_dir, query_yes_no
from cltk.data.fetch import FetchCorpus
from cltk.core.exceptions import CLTKException

class LatinTesseraeCorpusReader(TesseraeCorpusReader):
    """
    A corpus reader for Latin texts from the Tesserae-CLTK corpus
    """

    def __init__(self, root: str = None, fileids: str = r'.*\.tess', encoding: str = 'utf-8', lang: str = 'lat',
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
        self.corpus = "lat_text_tesserae"
        self._root = root

        self.__check_corpus()

        
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

        
        TesseraeCorpusReader.__init__(self, self.root, fileids, encoding, self.lang,
                                      word_tokenizer=self.word_tokenizer,
                                      sent_tokenizer=self.sent_tokenizer)

    @property
    def root(self):
        if not self._root:
            self._root = os.path.join(
                            get_cltk_data_dir(),
                            f"{self.lang}/text/{self.corpus}/texts")
        return self._root

    def __check_corpus(self):
        if not os.path.isdir(self.root):
            if self.root != os.path.join(
                    get_cltk_data_dir(),
                    f"{self.lang}/text/{self.corpus}/texts"):
                raise CLTKException(
                    f"Failed to instantiate LatinTesseraeCorpusReader. Root folder not found."
                )                        
            else:
                print(  # pragma: no cover
                    f"CLTK message: Unless a path is specifically passed to the 'root' parameter, this corpus reader expects to find the CLTK-Tesserae texts at {f'{self.lang}/text/{self.lang}_text_tesserae/texts'}."
                )  # pragma: no cover
                dl_is_allowed = query_yes_no(
                    f"Do you want to download CLTK-Tesserae Latin files?"
                )  # type: bool
                if dl_is_allowed:
                    fetch_corpus = FetchCorpus(language=self.lang)
                    fetch_corpus.import_corpus(
                        corpus_name=self.corpus
                    )
                else:
                    raise CLTKException(
                        f"Failed to instantiate LatinTesseraeCorpusReader. Rerun with 'root' parameter set to folder with .tess files or download the corpus to the CLTK_DATA folder."
                    )        

    def pos_sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        for sent in self.sents(fileids):
            data = self.nlp.analyze(text=sent)
            pos_sent = []
            for item in data:
                pos_sent.append(f"{item.string}/{item.upos}")
            yield pos_sent

    def tokenized_paras(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        # There is no para methods at present in Tess; use texts instead
        # to follow example in Bengfort et al. p. 49
        # TODO: Add lemmas
        for text in self.texts(fileids):
            tokenized_para = []
            sents = self.sent_tokenizer.tokenize(text)
            for sent in sents:
                tokenized_sent = []
                data = self.nlp.analyze(text=sent)
                for item in data:
                    tokenized_sent.append((item.string, item.upos))
                tokenized_para.append(tokenized_sent)
            yield tokenized_para
