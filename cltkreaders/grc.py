"""Corpus readers to support Greek text collections for use with CLTK; see readers.py for more information
"""

__author__ = [
    "Patrick J. Burns <patrick@diyclassics.org>",
]
__license__ = "MIT License."

import os.path
import codecs
from typing import Callable, Iterator, Union

import re
from natsort import natsorted

from cltkreaders.readers import CLTKPlaintextCorpusReader
from cltkreaders.readers import TesseraeCorpusReader, PerseusTreebankCorpusReader

from cltk import NLP
from cltk.core.data_types import Pipeline
from cltk.languages.utils import get_lang
from cltk.alphabet.processes import GreekNormalizeProcess
from cltk.dependency.processes import GreekStanzaProcess
from cltk.sentence.grc import GreekRegexSentenceTokenizer
from cltk.tokenizers.word import PunktWordTokenizer as GreekWordTokenizer
from cltk.lemmatize.grc import GreekBackoffLemmatizer

from cltk.utils import get_cltk_data_dir, query_yes_no
from cltk.data.fetch import FetchCorpus
from cltk.core.exceptions import CLTKException


class cltk_pos_tagger:
    def __init__(self, lang):
        self.lang = lang

    def tag(self, doc):
        pipeline = Pipeline(
            description="Greek pipeline for CLTK readers",
            processes=[GreekNormalizeProcess, GreekStanzaProcess],
            language=get_lang(self.lang),
        )

        cltk_nlp = NLP(
            language=self.lang, custom_pipeline=pipeline, suppress_banner=True
        )
        doc = cltk_nlp(doc)
        return [token.pos.name for token in doc]


class CLTKGreekCorpusReaderMixin:
    """Mixin class for CLTK Greek corpus readers"""

    # TODO: fileids should be in readers with in a non-language-specific mixin
    def fileids(self, match=None, **kwargs):
        """
        Return a list of file identifiers for the fileids that make up
        this corpus.
        """

        def get_metadata(metadata, key, value):
            return metadata.get(
                key, ""
            ).lower() == value.lower() or value.lower() in metadata.get(
                key, ""
            ).lower().split(
                "|"
            )

        fileids = self._fileids

        facets = []

        if match:
            match_fileids = [
                fileid
                for fileid in fileids
                if re.search(match, fileid, flags=re.IGNORECASE)
            ]
            facets.append(match_fileids)

        if kwargs:
            valid_keys = [list(k.keys()) for k in self._metadata.values()]
            valid_keys = sorted(
                list(set([item for sublist in valid_keys for item in sublist]))
            )
            for key, value in kwargs.items():
                value = str(value)
                if key not in valid_keys and key != "min_date" and key != "max_date":
                    raise ValueError(
                        f"Invalid key '{key}'. Valid keys are: {valid_keys}"
                    )
                if key != "min_date" and key != "max_date":
                    kwargs_fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if get_metadata(metadata, key, value)
                    ]
                    facets.append(kwargs_fileids)

        if "min_date" in kwargs or "max_date" in kwargs:
            if "date" in kwargs:
                print(
                    "Warning: You can only select min/max dates or a specific date. Using specific date."
                )
            else:
                if "min_date" in kwargs and "max_date" in kwargs:
                    date_fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"]) >= int(kwargs["min_date"])
                        and int(metadata["date"]) <= int(kwargs["max_date"])
                    ]
                elif "min_date" in kwargs:
                    date_fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"]) >= int(kwargs["min_date"])
                    ]
                elif "max_date" in kwargs:
                    date_fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"]) <= int(kwargs["max_date"])
                    ]
                facets.append(date_fileids)

        if facets:
            fileids = natsorted(list(set.intersection(*map(set, facets))))

        return fileids

    def _setup_greek_tools(self, nlp):
        if not self.word_tokenizer:
            self.word_tokenizer = GreekWordTokenizer()

        if not self.sent_tokenizer:
            self.sent_tokenizer = GreekRegexSentenceTokenizer()

        if not self.lemmatizer:
            self.lemmatizer = GreekBackoffLemmatizer()

        if not self.pos_tagger:
            self.pos_tagger = cltk_pos_tagger(lang=self.lang)


class GreekTesseraeCorpusReader(CLTKGreekCorpusReaderMixin, TesseraeCorpusReader):
    """
    A corpus reader for Greek texts from the Tesserae-CLTK corpus
    """

    def __init__(self, nlp=None, word_tokenizer=None, sent_tokenizer=None):
        self.lang = "grc"
        self.corpus = "grc_text_tesserae"
        self._root = self.root
        self.__check_corpus()

        if not word_tokenizer:
            self.word_tokenizer = GreekWordTokenizer()
        else:
            self.word_tokenizer = word_tokenizer

        if not sent_tokenizer:
            self.sent_tokenizer = GreekRegexSentenceTokenizer()
        else:
            self.sent_tokenizer = sent_tokenizer

        pipeline = Pipeline(
            description="Greek pipeline for Tesserae readers",
            processes=[GreekNormalizeProcess, GreekStanzaProcess],
            language=get_lang(self.lang),
        )
        self.nlp = NLP(
            language=self.lang, custom_pipeline=pipeline, suppress_banner=True
        )

        super().__init__(
            root=self._root,
            fileids=r".*\.tess",
            encoding="utf-8",
            lang="lat",
            normalization_form="NFC",
            word_tokenizer=word_tokenizer,
            sent_tokenizer=sent_tokenizer,
        )

    @property
    def root(self):
        return os.path.join(
            get_cltk_data_dir(), f"{self.lang}/text/{self.corpus}/texts"
        )

    def __check_corpus(self):
        if not os.path.isdir(self.root):
            if self.root != os.path.join(
                get_cltk_data_dir(), f"{self.lang}/text/{self.corpus}/texts"
            ):
                raise CLTKException(
                    "Failed to instantiate GreekTesseraeCorpusReader. Root folder not found."
                )
            else:
                print(  # pragma: no cover
                    f"CLTK message: Unless a path is specifically passed to the 'root' parameter, this corpus reader expects to find the CLTK-Tesserae texts at {f'{self.lang}/text/{self.lang}_text_tesserae/texts'}."
                )  # pragma: no cover
                dl_is_allowed = query_yes_no(
                    "Do you want to download CLTK-Tesserae Greek files?"
                )  # type: bool
                if dl_is_allowed:
                    fetch_corpus = FetchCorpus(language=self.lang)
                    fetch_corpus.import_corpus(corpus_name=self.corpus)
                    fetch_corpus.import_corpus(corpus_name=f"{self.lang}_models_cltk")
                else:
                    raise CLTKException(
                        "Failed to instantiate GreekTesseraeCorpusReader. Rerun with 'root' parameter set to folder with .tess files or download the corpus to the CLTK_DATA folder."
                    )

    def docs(self, fileids: Union[list, str] = None) -> Iterator[str]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :yield: Plaintext content of file
        """
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, "r", encoding=encoding) as f:
                doc = f.read()
                doc = doc.strip()
                yield doc

    def pos_sents(
        self, fileids: Union[list, str] = None, preprocess: Callable = None
    ) -> Iterator[list]:
        for sent in self.sents(fileids, preprocess=preprocess):
            data = self.nlp.analyze(text=sent)
            pos_sent = []
            for item in data:
                pos_sent.append(f"{item.string}/{item.upos}")
            yield pos_sent

    def words(
        self, fileids: Union[list, str] = None, preprocess: Callable = None
    ) -> Iterator[list]:
        for pos_sent in self.pos_sents(fileids, preprocess=preprocess):
            for item in pos_sent:
                yield item.split("/")[0]


# TODO: Add corpus download support following Tesserae example
GreekPerseusTreebankCorpusReader = PerseusTreebankCorpusReader


class GreekPlaintextCorpusReader(CLTKGreekCorpusReaderMixin, CLTKPlaintextCorpusReader):
    """
    A corpus reader for Ancient Greek texts
    """

    def __init__(
        self,
        root=None,
        fileids=r".*\.txt",
        encoding="utf8",
        lang="grc",
        nlp=None,
        word_tokenizer=None,
        sent_tokenizer=None,
        lemmatizer=None,
        pos_tagger=None,
    ):
        self._root = root
        self.lang = lang
        self.nlp = nlp
        self.word_tokenizer = word_tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.lemmatizer = lemmatizer
        self.pos_tagger = pos_tagger

        # TODO: Hold until spaCy refactoring
        # Doc.set_extension("metadata", default=None, force=True)
        # Span.set_extension("metadata", default=None, force=True)
        # Token.set_extension("metadata", default=None, force=True)
        self._setup_greek_tools(self.nlp)
        CLTKPlaintextCorpusReader.__init__(self, root, fileids, encoding)

    def texts(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
    ) -> Iterator[Union[str, object]]:
        for doc in self.docs(
            fileids,
        ):
            yield doc

    def sents(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        plaintext: bool = False,
    ) -> Iterator[Union[str, object]]:
        for doc in self.docs(
            fileids,
        ):
            sents = self.sent_tokenizer.tokenize(doc)
            for sent in sents:
                yield sent.strip()

    def tokens(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        plaintext: bool = False,
    ) -> Iterator[Union[str, object]]:
        for sent in self.sents(
            fileids,
            preprocess=preprocess,
            plaintext=False,
        ):
            for sent in self.sents(fileids, preprocess=preprocess):
                tokens = self.word_tokenizer.tokenize(sent)
                for token in tokens:
                    yield token

    def tokenized_sents(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
    ) -> Iterator[Union[str, object]]:
        for sent in self.sents(
            fileids,
            preprocess=preprocess,
            plaintext=False,
        ):
            tokenized_sent = []
            tokens = self.word_tokenizer.tokenize(sent)
            lemmas = [lemma for _, lemma in self.lemmatizer.lemmatize(tokens)]
            # TODO: Implement POS tagging
            postags = ["N/A" for _ in tokens]

            for token, lemma, postag in zip(tokens, lemmas, postags):
                tokenized_sent.append((token, lemma, postag))
            yield tokenized_sent

    def chunks(
        self,
        fileids: Union[str, list] = None,
        chunk_size: int = 1000,
        basis="token",
        keep_tail: bool = True,
        preprocess: Callable = None,
    ) -> Iterator[Union[str, object]]:
        chunks = []
        chunk = []
        counter = 0

        for sent in self.sents(fileids):
            chunk.append(sent)
            if basis == "char":
                counter += len(sent)
            else:
                counter += len(self.word_tokenizer.tokenize(sent))

            if counter > chunk_size:
                chunks.append(chunk)
                counter = 0
                chunk = []

        if len(chunks) == 0:
            chunks.append(chunk)

        if keep_tail and len(chunks) > 1:
            chunks.append(chunk)
            chunks = chunks[:-2] + [chunks[-2] + chunks[-1]]
        elif len(chunks) > 1:
            chunks = chunks[:-1]
        else:
            pass

        for chunk in chunks:
            yield chunk
