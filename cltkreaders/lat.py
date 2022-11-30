"""Corpus readers to support Latin text collections for use with CLTK; see readers.py for more information
"""

__author__ = ["Patrick J. Burns <patrick@diyclassics.org>",]
__license__ = "MIT License."

import os.path
from typing import Callable, Iterator, Union

import spacy
nlp = spacy.load('la_core_cltk_sm')
nlp.max_length = 2500000

from cltkreaders.readers import CLTKPlaintextCorpusReader
from cltkreaders.readers import TesseraeCorpusReader, PerseusTreebankCorpusReader, PerseusCorpusReader

from cltk import NLP
from cltk.core.data_types import Pipeline
from cltk.languages.utils import get_lang
from cltk.alphabet.processes import LatinNormalizeProcess
from cltk.dependency.processes import LatinStanzaProcess
from cltk.sentence.lat import LatinPunktSentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer
from cltk.lemmatize.lat import LatinBackoffLemmatizer

from cltk.utils import get_cltk_data_dir, query_yes_no
from cltk.data.fetch import FetchCorpus
from cltk.core.exceptions import CLTKException

class spacy_segmenter():
    def __init__(self):
        pass
    def tokenize(self, doc):
        doc = nlp(doc)
        return [sent for sent in doc.sents]

class spacy_tokenizer():
    def __init__(self):
        pass

    def tokenize(self, doc):
        if isinstance(doc, str):
            doc = nlp(doc)
        return [token for token in doc]

class spacy_lemmatizer():
    def __init__(self):
        pass    
    def lemmatize(self, doc):
        return [token.lemma_ for token in doc]

class spacy_pos_tagger():
    def __init__(self):
        pass
    def tag(self, doc):
        return [token.pos_ for token in doc]

class cltk_pos_tagger():
    def __init__(self, lang):
        self.lang = lang
    def tag(self, doc):
        pipeline = Pipeline(description="Latin pipeline for CLTK readers", 
                                processes=[LatinNormalizeProcess, LatinStanzaProcess], 
                                language=get_lang(self.lang))
            
        cltk_nlp = NLP(language=self.lang, custom_pipeline=pipeline, suppress_banner=True)
        doc = cltk_nlp(doc)
        return [token.pos.name for token in doc]

class CLTKLatinCorpusReaderMixin():

    def _setup_latin_tools(self, nlp):
        if nlp == 'spacy':
            self.sent_tokenizer = spacy_segmenter()
            self.word_tokenizer = spacy_tokenizer()
            self.lemmatizer = spacy_lemmatizer()
            self.pos_tagger = spacy_pos_tagger()
        else:        
            if not word_tokenizer:
                self.word_tokenizer = LatinWordTokenizer()
            else:
                self.word_tokenizer = word_tokenizer

            if not sent_tokenizer:
                self.sent_tokenizer = LatinPunktSentenceTokenizer()
            else:
                self.sent_tokenizer = sent_tokenizer

            if lemmatizer:
                self.lemmatizer = lemmatizer
            else:
                self.lemmatizer = LatinBackoffLemmatizer()                    

            if pos_tagger:
                self.pos_tagger = pos_tagger
            else:
                self.pos_tagger = cltk_pos_tagger(lang=self.lang)          


    def sents(self, fileids: Union[list, str] = None, unline: bool = True, preprocess: Callable = None) -> Iterator[list]:
        for para in self.paras(fileids):
            if unline:
                para = ' '.join(para.split()).strip()
            sents = self.sent_tokenizer.tokenize(para)
            for sent in sents:
                if preprocess:
                    if self.nlp == 'spacy':
                        sent = preprocess(sent.text)
                    else:
                        sent = preprocess(sent)
                yield sent      

    def words(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        print(self.nlp)
        for sent in self.sents(fileids, preprocess=preprocess):
            words = self.word_tokenizer.tokenize(sent)
            for word in words:
                yield word.text            

    def tokenized_paras(self, fileids: Union[list, str] = None, unline: bool = True, preprocess: Callable = None) -> Iterator[list]:
        for para in self.paras(fileids):
            tokenized_para = []
            if unline:
                para = ' '.join(para.split()).strip()
            sents = self.sent_tokenizer.tokenize(para)
            for sent in sents:
                if preprocess:
                    if self.nlp == 'spacy':
                        sent = preprocess(sent.text)
                    else:
                        sent = preprocess(sent)
                if self.nlp == 'spacy':
                    tokens_ = [token for token in self.word_tokenizer.tokenize(sent)]
                    words = [token.text for token in tokens_]
                    lemmas = [token.lemma_ for token in tokens_]
                    postags = [token.pos_ for token in tokens_]
                else:
                    words = [token for token in self.word_tokenizer.tokenize(sent)]
                    lemmas = [lemma for _, lemma in self.lemmatizer.lemmatize(sent)]
                    postags = [postag for postag in self.pos_tagger.tag(sent)]
                tokenized_para.append(list(zip(words, lemmas, postags)))
            yield tokenized_para
        
    def tokenized_sents(self, fileids: Union[list, str] = None, unline: bool = True, preprocess: Callable = None) -> Iterator[list]:
        for sent in self.tokenized_paras(fileids, unline=unline, preprocess=preprocess):
            yield sent

class LatinTesseraeCorpusReader(CLTKLatinCorpusReaderMixin, TesseraeCorpusReader):
    """
    A corpus reader for Latin texts from the Tesserae-CLTK corpus
    """

    def __init__(self, root: str = None, fileids: str = r'.*\.tess', encoding: str = 'utf-8', lang: str = 'lat',
                 nlp: str = 'spacy', normalization_form: str = 'NFC',
                word_tokenizer: Callable = None, sent_tokenizer: Callable = None, 
                lemmatizer: Callable = None, pos_tagger: Callable = None,
                **kwargs):
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

        self.nlp = nlp
        self._setup_latin_tools(self.nlp)

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
                    f"Failed to instantiate corpus reader. Root folder not found."
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
                    fetch_corpus.import_corpus(corpus_name=f'{self.lang}_models_cltk')
                else:
                    raise CLTKException(
                        f"Failed to instantiate corpus reader. Rerun with 'root' parameter set to folder with corpus files or download the corpus to the CLTK_DATA folder."
                    )                                  


# TODO: Add corpus download support following Tesserae example
LatinPerseusTreebankCorpusReader = PerseusTreebankCorpusReader


class LatinPerseusCorpusReader(CLTKLatinCorpusReaderMixin, PerseusCorpusReader):
    """
    A corpus reader for working Perseus XML files, inc.
    PDILL: https://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:PDILL

    NB: `root` should point to a directory containing the xml files
    """

    def __init__(self, root: str, fileids: str = r'.*\.xml', encoding: str = 'utf8', ns = None,lang='la', nlp='spacy',
                word_tokenizer: Callable = None, sent_tokenizer: Callable = None, 
                lemmatizer: Callable = None, pos_tagger: Callable = None,
                **kwargs):
        
        self.nlp = nlp
        self._setup_latin_tools(self.nlp)                                 
        PerseusCorpusReader.__init__(self, root, fileids, encoding=encoding, nlp=nlp, ns=ns)                


class CSELCorpusReader(LatinPerseusCorpusReader):
    """
    A corpus reader for working Perseus CSEL XML files, inc.
    cf. https://github.com/OpenGreekAndLatin/csel-dev
    
    NB: `root` should point to a directory containing the xml files
    """

    def __init__(self, root: str, fileids: str = r'.*\.xml', encoding: str = 'utf8', lang='la', 
                ns={'tei': 'http://www.tei-c.org/ns/1.0'}, nlp='spacy',
                word_tokenizer: Callable = None, sent_tokenizer: Callable = None, 
                lemmatizer: Callable = None, pos_tagger: Callable = None,
                **kwargs):                             

        LatinPerseusCorpusReader.__init__(self, root, fileids, encoding=encoding, nlp=nlp, ns=ns)
        

class LatinLibraryCorpusReader(CLTKLatinCorpusReaderMixin, CLTKPlaintextCorpusReader):
    """
    A corpus reader for Latin texts from the Latin Library
    """

    def __init__(self, root: str = None, fileids: str = r'.*\.txt', encoding: str = 'utf-8', lang: str = 'lat',
                 nlp: str = 'spacy', normalization_form: str = 'NFC',
                 word_tokenizer: Callable = None, sent_tokenizer: Callable = None, 
                 lemmatizer: Callable = None, pos_tagger: Callable = None,
                 **kwargs):
        self.lang = lang
        self.corpus = "lat_text_latin_library"
        self._root = root

        self.__check_corpus()

        self.nlp = nlp
        self._setup_latin_tools(self.nlp)                 

        CLTKPlaintextCorpusReader.__init__(self, self.root, fileids, encoding, self.lang)

    @property
    def root(self):
        if not self._root:
            self._root = os.path.join(
                            get_cltk_data_dir(),
                            f"{self.lang}/text/{self.corpus}")
        return self._root

    def __check_corpus(self):
        if not os.path.isdir(self.root):
            if self.root != os.path.join(
                    get_cltk_data_dir(),
                    f"{self.lang}/text/{self.corpus}"):
                raise CLTKException(
                    f"Failed to instantiate corpus reader. Root folder not found."
                )                        
            else:
                print(  # pragma: no cover
                    f"CLTK message: Unless a path is specifically passed to the 'root' parameter, this corpus reader expects to find the files at {f'{self.lang}/text/{self.lang}_text_latin_library'}."
                )  # pragma: no cover
                dl_is_allowed = query_yes_no(
                    f"Do you want to download {self.corpus} corpus files?"
                )  # type: bool
                if dl_is_allowed:
                    fetch_corpus = FetchCorpus(language=self.lang)
                    fetch_corpus.import_corpus(
                        corpus_name=self.corpus
                    )
                    fetch_corpus.import_corpus(corpus_name=f'{self.lang}_models_cltk')
                else:
                    raise CLTKException(
                        f"Failed to instantiate corpus reader. Rerun with 'root' parameter set to folder with corpus files or download the corpus to the CLTK_DATA folder."
                    )               
