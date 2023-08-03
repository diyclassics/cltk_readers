"""Corpus readers to support Latin text collections for use with CLTK; see readers.py for more information
"""

__author__ = [
    "Patrick J. Burns <patrick@diyclassics.org>",
]
__license__ = "MIT License."

import os.path
import codecs
from typing import Callable, Iterator, Union, List

import re
from lxml import etree
from collections import defaultdict

from cltkreaders.readers import CLTKPlaintextCorpusReader
from cltkreaders.readers import (
    TesseraeCorpusReader,
    PerseusTreebankCorpusReader,
    PerseusCorpusReader,
)

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

import spacy
from spacy.tokens import Doc, Span, Token
import textacy


class cltk_pos_tagger:
    def __init__(self, lang):
        self.lang = lang

    def tag(self, doc):
        pipeline = Pipeline(
            description="Latin pipeline for CLTK readers",
            processes=[LatinNormalizeProcess, LatinStanzaProcess],
            language=get_lang(self.lang),
        )

        cltk_nlp = NLP(
            language=self.lang, custom_pipeline=pipeline, suppress_banner=True
        )
        doc = cltk_nlp(doc)
        return [token.pos.name for token in doc]


class CLTKLatinCorpusReaderMixin:
    """Mixin class for CLTK Latin corpus readers"""

    def fileids(self, match=None, **kwargs):
        """
        Return a list of file identifiers for the fileids that make up
        this corpus.
        """

        fileids = self._fileids

        if match:
            fileids = [
                fileid
                for fileid in fileids
                if re.search(match, fileid, flags=re.IGNORECASE)
            ]

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
                    fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if metadata.get(key, "").lower() == value.lower()
                        and fileid in fileids
                    ]

        if "min_date" in kwargs or "max_date" in kwargs:
            if "date" in kwargs:
                print(
                    "Warning: You can only select min/max dates or a specific date. Using specific date."
                )
            else:
                if "min_date" in kwargs and "max_date" in kwargs:
                    fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"]) >= int(kwargs["min_date"])
                        and int(metadata["date"]) <= int(kwargs["max_date"])
                        and fileid in fileids
                    ]
                elif "min_date" in kwargs:
                    fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"]) >= int(kwargs["min_date"])
                        and fileid in fileids
                    ]
                elif "max_date" in kwargs:
                    fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"])
                        <= int(kwargs["max_date"] and fileid in fileids)
                    ]
        self._fileids = fileids
        return self._fileids

    def _setup_latin_tools(self, nlp):
        if nlp.startswith("la_") and nlp.count("_") == 3:
            self.model = spacy.load(nlp)
            self.model.max_length = 2500000
        elif nlp == "spacy":
            self.model = spacy.load("la_core_web_lg")
            self.model.max_length = 2500000

            class spacy_segmenter:
                def __init__(self):
                    pass

                def tokenize(self, doc):
                    doc = self.model(doc)
                    return [sent for sent in doc.sents]

            class spacy_tokenizer:
                def __init__(self):
                    pass

                def tokenize(self, doc):
                    if isinstance(doc, str):
                        doc = self.model(doc)
                    return [token for token in doc]

            class spacy_lemmatizer:
                def __init__(self):
                    pass

                def lemmatize(self, doc):
                    return [token.lemma_ for token in doc]

            class spacy_pos_tagger:
                def __init__(self):
                    pass

                def tag(self, doc):
                    return [token.pos_ for token in doc]

            self.sent_tokenizer = spacy_segmenter()
            self.word_tokenizer = spacy_tokenizer()
            self.lemmatizer = spacy_lemmatizer()
            self.pos_tagger = spacy_pos_tagger()
        else:
            if not self.word_tokenizer:
                self.word_tokenizer = LatinWordTokenizer()

            if not self.sent_tokenizer:
                self.sent_tokenizer = LatinPunktSentenceTokenizer()

            if not self.lemmatizer:
                self.lemmatizer = LatinBackoffLemmatizer()

            if not self.pos_tagger:
                self.pos_tagger = cltk_pos_tagger(lang=self.lang)

    def sents(
        self,
        fileids: Union[list, str] = None,
        unline: bool = True,
        preprocess: Callable = None,
    ) -> Iterator[list]:
        for para in self.paras(fileids):
            if unline:
                para = " ".join(para.split()).strip()
            sents = self.sent_tokenizer.tokenize(para)
            for sent in sents:
                if preprocess:
                    if self.nlp == "spacy":
                        sent = preprocess(sent.text)
                    else:
                        sent = preprocess(sent)
                yield sent

    def words(
        self, fileids: Union[list, str] = None, preprocess: Callable = None
    ) -> Iterator[list]:
        for sent in self.sents(fileids, preprocess=preprocess):
            words = self.word_tokenizer.tokenize(sent)
            for word in words:
                yield word.text

    def tokenized_paras(
        self,
        fileids: Union[list, str] = None,
        unline: bool = True,
        preprocess: Callable = None,
    ) -> Iterator[list]:
        for para in self.paras(fileids):
            tokenized_para = []
            if unline:
                para = " ".join(para.split()).strip()
            sents = self.sent_tokenizer.tokenize(para)
            for sent in sents:
                if preprocess:
                    if self.nlp == "spacy":
                        sent = preprocess(sent.text)
                    else:
                        sent = preprocess(sent)
                if self.nlp == "spacy":
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

    def tokenized_sents(
        self,
        fileids: Union[list, str] = None,
        unline: bool = True,
        preprocess: Callable = None,
        simple=False,
    ) -> Iterator[list]:
        for para in self.tokenized_paras(fileids, unline=unline, preprocess=preprocess):
            for sent in para:
                if simple:
                    yield [token for token, _, _ in sent]
                else:
                    yield sent

    def pos_sents(
        self, fileids: Union[list, str] = None, preprocess: Callable = None
    ) -> Iterator[list]:
        for sent in self.tokenized_sents(fileids, preprocess=preprocess):
            yield ["/".join([word, postag]) for word, _, postag in sent]


class LatinTesseraeCorpusReader(CLTKLatinCorpusReaderMixin, TesseraeCorpusReader):
    def __init__(self, nlp=None):
        self.lang = "lat"
        self.corpus = "lat_text_tesserae"
        self._root = self.root
        self.__check_corpus()
        super().__init__(
            root=self._root, fileids=r".*\.tess", encoding="utf-8", lang="lat"
        )
        if not nlp:
            self.nlp = "la_core_web_lg"
        else:
            self.nlp = "spacy"

        Doc.set_extension("metadata", default=None, force=True)
        Span.set_extension("sentence_citation", default=False, force=True)
        Span.set_extension("citation", default=False, force=True)
        Span.set_extension("metadata", default=None, force=True)
        Token.set_extension("metadata", default=None, force=True)
        Token.set_extension("citation", default=False, force=True)

        self._setup_latin_tools(self.nlp)

    @property
    def root(self):
        return os.path.join(
            get_cltk_data_dir(), f"{self.lang}/text/{self.corpus}/texts"
        )

    def __check_corpus(self):
        if not os.path.isdir(self._root):
            if self._root != os.path.join(
                get_cltk_data_dir(), f"{self.lang}/text/{self.corpus}/texts"
            ):
                raise CLTKException(
                    f"Failed to instantiate corpus reader. Root folder not found."
                )
            else:
                print(
                    f"CLTK message: Unless a path is specifically passed to the 'root' parameter, this corpus reader expects to find the CLTK-Tesserae texts at {f'{self.lang}/text/{self.lang}_text_tesserae/texts'}."
                )
                dl_is_allowed = query_yes_no(
                    f"Do you want to download CLTK-Tesserae Latin files?"
                )  # type: bool
                if dl_is_allowed:
                    fetch_corpus = FetchCorpus(language=self.lang)
                    fetch_corpus.import_corpus(corpus_name=self.corpus)
                    fetch_corpus.import_corpus(corpus_name=f"{self.lang}_models_cltk")
                else:
                    raise CLTKException(
                        f"Failed to instantiate corpus reader. Rerun with 'root' parameter set to folder with corpus files or download the corpus to the CLTK_DATA folder."
                    )

    def fileids(self, match=None, **kwargs):
        """
        Return a list of file identifiers for the fileids that make up
        this corpus.
        """

        fileids = self._fileids

        if match:
            fileids = [
                fileid
                for fileid in fileids
                if re.search(match, fileid, flags=re.IGNORECASE)
            ]

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
                    fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if metadata.get(key, "").lower() == value.lower()
                        and fileid in fileids
                    ]

        if "min_date" in kwargs or "max_date" in kwargs:
            if "date" in kwargs:
                print(
                    "Warning: You can only select min/max dates or a specific date. Using specific date."
                )
            else:
                if "min_date" in kwargs and "max_date" in kwargs:
                    fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"]) >= int(kwargs["min_date"])
                        and int(metadata["date"]) <= int(kwargs["max_date"])
                        and fileid in fileids
                    ]
                elif "min_date" in kwargs:
                    fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"]) >= int(kwargs["min_date"])
                        and fileid in fileids
                    ]
                elif "max_date" in kwargs:
                    fileids = [
                        fileid
                        for fileid, metadata in self._metadata.items()
                        if int(metadata["date"])
                        <= int(kwargs["max_date"] and fileid in fileids)
                    ]
        self._fileids = fileids
        return self._fileids

    def spacy_docs(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        line_citations: bool = True,
        sent_citations: bool = True,
    ) -> Iterator[object]:
        def make_line_spans(lens: list) -> List[tuple]:
            spans = []
            begin = 0
            for i in range(len(lens)):
                end = begin + lens[i]
                spans.append((begin, end))
                begin = end
            return spans

        def getIndex(lst: list, num: int) -> int:
            # cf. https://stackoverflow.com/a/61158780
            # But +1 for inclusive ranges, i.e. sometimes sentences end on last word of line.
            for idx, val in enumerate(lst):
                if num in range(val[0], val[1] + 1):
                    return idx
            return -1

        for path, encoding in self.abspaths(fileids, include_encoding=True):
            current_file = os.path.basename(path)

            with codecs.open(path, "r", encoding=encoding) as f:
                doc = f.read()
                doc = doc.strip()

            doc_rows = defaultdict(str)

            # Tesserae files are generally divided such that there are no line breaks within citation sections; there
            # are exceptions to protect against, i.e. remove \n when not followed by the citation indicator (<)
            doc = re.sub(r"\n([^<])", r" \g<1>", doc)

            lines = [line for line in doc.split("\n") if line]
            for line in lines:
                try:
                    k, v = line.split(">", 1)
                    k = f"{k}>"
                    v = v.strip()
                    if preprocess:
                        v = preprocess(v)
                    v = self.model.make_doc(v)
                    doc_rows[k] = v
                except:
                    print(
                        f"The following line is not formatted corrected and has been skipped: {line}\n"
                    )

            # doc_rows = dict(rows)

            citations = list(doc_rows.keys())
            lines = list(doc_rows.values())
            lens = [len(line) for line in lines]
            line_spans = make_line_spans(lens)
            doc = Doc.from_docs(lines)
            metadata = self._metadata[current_file]
            spacy_doc = textacy.make_spacy_doc((doc.text, metadata), lang=self.model)

            if line_citations:
                spacy_doc.spans["lines"] = [
                    Span(spacy_doc, span[0], span[1], label="line")
                    for span in line_spans
                ]
                for cit, line in zip(citations, spacy_doc.spans["lines"]):
                    line._.citation = cit
                    line._.metadata = spacy_doc._.meta

            if sent_citations:
                citation_ranges = []
                citation_ranges_formatted = []

                citation_start = 0

                for sent in spacy_doc.sents:
                    sent_end = sent.end
                    citation_end = getIndex(line_spans, sent_end)
                    citation_ranges.append((citation_start, citation_end))
                    citation_ranges_formatted.append(
                        (citations[citation_start], citations[citation_end])
                    )
                    if sent_end == line_spans[citation_end][1]:
                        citation_start = citation_end + 1
                    else:
                        citation_start = citation_end

                # Add sentence citations to sents
                for cit, sent in zip(citation_ranges_formatted, spacy_doc.sents):
                    sent._.sentence_citation = cit
                    sent._.metadata = spacy_doc._.meta

            yield spacy_doc

    def lines(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        line_citations: bool = True,
        sent_citations: bool = True,
        plaintext: bool = False,
    ) -> Iterator[Union[str, object]]:

        for doc in self.spacy_docs(
            fileids,
            preprocess=preprocess,
            line_citations=line_citations,
            sent_citations=sent_citations,
        ):
            for line in doc.spans["lines"]:
                if plaintext:
                    yield line.text
                else:
                    yield line

    def doc_rows(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        line_citations: bool = True,
        sent_citations: bool = True,
        plaintext: bool = False,
    ) -> Iterator[dict]:
        for doc in self.spacy_docs(
            fileids,
            preprocess=preprocess,
            line_citations=line_citations,
            sent_citations=sent_citations,
        ):
            if plaintext:
                yield {line._.citation: line.text for line in doc.spans["lines"]}
            else:
                yield {line._.citation: line for line in doc.spans["lines"]}

    doc_dicts = doc_rows

    def sents(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        line_citations: bool = True,
        sent_citations: bool = True,
        plaintext: bool = False,
    ) -> Iterator[Union[str, object]]:
        for doc in self.spacy_docs(
            fileids,
            preprocess=preprocess,
            line_citations=line_citations,
            sent_citations=sent_citations,
        ):
            for sent in doc.sents:
                if plaintext:
                    yield sent.text
                else:
                    yield sent

    def tokens(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        line_citations: bool = True,
        sent_citations: bool = True,
        plaintext: bool = False,
    ) -> Iterator[Union[str, object]]:
        for line in self.lines(
            fileids,
            preprocess=preprocess,
            line_citations=line_citations,
            sent_citations=sent_citations,
            plaintext=False,
        ):
            for i, token in enumerate(line):
                token._.citation = (line._.citation, i)
                token._.metadata = line._.metadata
                if plaintext:
                    yield token.text
                else:
                    yield token

    def tokenized_sents(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        line_citations: bool = True,
        sent_citations: bool = True,
    ) -> Iterator[Union[str, object]]:
        for sent in self.sents(
            fileids,
            preprocess=preprocess,
            line_citations=line_citations,
            sent_citations=sent_citations,
            plaintext=False,
        ):
            tokenized_sent = []
            for token in sent:
                tokenized_sent.append((token.text, token.lemma_, token.pos_))
            yield tokenized_sent

    def pos_sents(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        line_citations: bool = True,
        sent_citations: bool = True,
    ) -> Iterator[Union[str, object]]:
        for sent in self.sents(
            fileids,
            preprocess=preprocess,
            line_citations=line_citations,
            sent_citations=sent_citations,
            plaintext=False,
        ):
            pos_sent = []
            for token in sent:
                pos_sent.append("/".join([token.text, token.pos_]))
            yield pos_sent

    def concordance(
        self,
        fileids: Union[str, list] = None,
        basis: str = "norm",
        only_alpha: bool = True,
        preprocess: Callable = None,
    ) -> dict:
        concordance_dict = defaultdict(list)
        for token in self.tokens(fileids, preprocess=preprocess):
            if only_alpha:
                if not token.is_alpha:
                    continue
            if basis == "norm":
                concordance_dict[token.norm_].append(token._.citation)
            elif basis == "lemma":
                concordance_dict[token.lemma_].append(token._.citation)
            else:
                concordance_dict[token.text].append(token._.citation)
        concordance_dict = {k: v for k, v in sorted(concordance_dict.items())}
        return concordance_dict

    def chunks(
        self,
        fileids: Union[str, list] = None,
        chunk_size: int = 1000,
        basis="token",
        keep_tail: bool = True,
        preprocess: Callable = None,
        line_citations: bool = True,
        sent_citations: bool = True,
    ) -> Iterator[Union[str, object]]:
        chunks = []
        chunk = []
        counter = 0

        for sent in self.sents(fileids):
            chunk.append(sent.as_doc())
            if basis == "char":
                counter += len(sent.text)
            else:
                counter += len(sent)

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

        chunks = [Doc.from_docs(chunk) for chunk in chunks]

        for chunk in chunks:
            # Assigns chunk metadata from the last sent; TODO: Fix so that it works across fileids
            chunk._.metadata = sent._.metadata
            yield chunk


# TODO: Add corpus download support following Tesserae example
LatinPerseusTreebankCorpusReader = PerseusTreebankCorpusReader


class LatinPerseusCorpusReader(CLTKLatinCorpusReaderMixin, PerseusCorpusReader):
    """
    A corpus reader for working Perseus XML files, inc.
    PDILL: https://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:PDILL

    NB: `root` should point to a directory containing the xml files
    """

    def __init__(
        self,
        root: str,
        fileids: str = r".*\.xml",
        encoding: str = "utf8",
        ns=None,
        lang="la",
        nlp="spacy",
        word_tokenizer: Callable = None,
        sent_tokenizer: Callable = None,
        lemmatizer: Callable = None,
        pos_tagger: Callable = None,
        **kwargs,
    ):

        self.word_tokenizer = word_tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.lemmatizer = lemmatizer
        self.pos_tagger = pos_tagger

        self.nlp = nlp
        self._setup_latin_tools(self.nlp)
        PerseusCorpusReader.__init__(
            self, root, fileids, encoding=encoding, nlp=nlp, ns=ns
        )


class LatinLibraryCorpusReader(CLTKLatinCorpusReaderMixin, CLTKPlaintextCorpusReader):
    """
    A corpus reader for Latin texts from the Latin Library
    """

    def __init__(self, nlp=None):
        # self.fileids = r".*\.txt"
        self.lang = "lat"
        self.corpus = "lat_text_latin_library"
        self.__check_corpus()
        CLTKPlaintextCorpusReader.__init__(self, self.root, r".*\.txt", "utf8")

        if not nlp:
            self.nlp = "la_core_web_lg"
        else:
            self.nlp = "spacy"

        Doc.set_extension("metadata", default=None, force=True)
        Span.set_extension("metadata", default=None, force=True)
        Token.set_extension("metadata", default=None, force=True)
        self._setup_latin_tools(self.nlp)

    @property
    def root(self):
        return os.path.join(get_cltk_data_dir(), f"{self.lang}/text/{self.corpus}")

    def __check_corpus(self):
        if not os.path.isdir(self.root):
            if self.root != os.path.join(
                get_cltk_data_dir(), f"{self.lang}/text/{self.corpus}"
            ):
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
                    fetch_corpus.import_corpus(corpus_name=self.corpus)
                    fetch_corpus.import_corpus(corpus_name=f"{self.lang}_models_cltk")
                else:
                    raise CLTKException(
                        f"Failed to instantiate corpus reader. Rerun with 'root' parameter set to folder with corpus files or download the corpus to the CLTK_DATA folder."
                    )

    def spacy_docs(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
    ) -> Iterator[object]:
        print(fileids)
        for i, (path, encoding) in enumerate(
            self.abspaths(fileids, include_encoding=True)
        ):
            if isinstance(fileids, str):
                current_file = fileids
            else:
                current_file = fileids[i]

            with codecs.open(path, "r", encoding=encoding) as f:
                doc = f.read()
                doc = doc.strip()

            # Paragraphize doc spacing
            doc = "\n".join(
                [para.strip() for para in doc.split("\n\n") if para.strip()]
            )

            metadata = self._metadata[current_file]
            spacy_doc = textacy.make_spacy_doc((doc, metadata), lang=self.model)
            yield spacy_doc

    def sents(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
        plaintext: bool = False,
    ) -> Iterator[Union[str, object]]:
        for doc in self.spacy_docs(
            fileids,
            preprocess=preprocess,
        ):
            for sent in doc.sents:
                sent._.metadata = doc._.meta
                if plaintext:
                    yield sent.text
                else:
                    yield sent

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
            for i, token in enumerate(sent):
                token._.metadata = sent._.metadata
                if plaintext:
                    yield token.text
                else:
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
            for token in sent:
                tokenized_sent.append((token.text, token.lemma_, token.pos_))
            yield tokenized_sent

    def pos_sents(
        self,
        fileids: Union[str, list] = None,
        preprocess: Callable = None,
    ) -> Iterator[Union[str, object]]:
        for sent in self.sents(
            fileids,
            preprocess=preprocess,
            plaintext=False,
        ):
            pos_sent = []
            for token in sent:
                pos_sent.append("/".join([token.text, token.pos_]))
            yield pos_sent

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
            chunk.append(sent.as_doc())
            if basis == "char":
                counter += len(sent.text)
            else:
                counter += len(sent)

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

        chunks = [Doc.from_docs(chunk) for chunk in chunks]

        for chunk in chunks:
            # Assigns chunk metadata from the last sent; TODO: Fix so that it works across fileids
            chunk._.metadata = sent._.metadata
            yield chunk


class CamenaCorpusReader(LatinPerseusCorpusReader, CLTKLatinCorpusReaderMixin):
    """
    A corpus reader for working with Camena TEI/XML docs
    cf. https://github.com/nevenjovanovic/camena-neolatinlit
    """

    def __init__(
        self,
        root: str,
        fileids: str = r".*\.xml",
        encoding: str = "utf8",
        lang="la",
        ns=None,
        nlp="spacy",
        include_front=True,
        word_tokenizer: Callable = None,
        sent_tokenizer: Callable = None,
        lemmatizer: Callable = None,
        pos_tagger: Callable = None,
        **kwargs,
    ):

        self.include_front = include_front

        LatinPerseusCorpusReader.__init__(
            self, root, fileids, encoding=encoding, nlp=nlp, ns=ns
        )

    def _get_xml_encoding_from_file(self, file):
        with open(file, "rb") as f:
            return re.search(rb'encoding="(.+?)"', f.read()).group(1).decode()

    def docs(self, fileids: Union[str, list] = None):
        """
        Returns the complete text of a .xml file, closing the document after
        we are done reading it and yielding it in a memory-safe fashion.
        """

        # Create a generator, loading one document into memory at a time.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            encoding = self._get_xml_encoding_from_file(path)
            with codecs.open(path, "r", encoding=encoding) as f:
                doc = f.read()
                if doc:
                    x = etree.fromstring(
                        bytes(doc, encoding="utf-8"),
                        parser=etree.XMLParser(huge_tree=True),
                    )
                    yield etree.tostring(x, pretty_print=True, encoding=str)

    def contents(self, fileids: Union[str, list] = None):
        for doc in self.docs(fileids):
            root = etree.fromstring(doc)
            if self.include_front:
                contents = root.xpath(".//*[self::front or self::body]")
            else:
                contents = root.findall(".//body")
            for content in contents:
                yield content

    def paras(self, fileids: Union[str, list] = None):
        for content in self.contents(fileids):
            paras = content.findall(".//p")

            for para in paras:
                yield " ".join(para.itertext())


class CSELCorpusReader(LatinPerseusCorpusReader, CLTKLatinCorpusReaderMixin):
    """
    A corpus reader for working with CSEL TEI/XML docs
    cf. TK
    """

    def __init__(
        self,
        root: str,
        fileids: str = r".*\.xml",
        encoding: str = "utf8",
        lang="la",
        ns={"tei": "http://www.tei-c.org/ns/1.0"},
        nlp="spacy",
        word_tokenizer: Callable = None,
        sent_tokenizer: Callable = None,
        lemmatizer: Callable = None,
        pos_tagger: Callable = None,
        **kwargs,
    ):

        LatinPerseusCorpusReader.__init__(
            self, root, fileids, encoding=encoding, nlp=nlp, ns=ns
        )

    def _get_xml_encoding_from_file(self, file):
        with open(file, "rb") as f:
            return re.search(rb'encoding="(.+?)"', f.read()).group(1).decode()

    def docs(self, fileids: Union[str, list] = None):
        """
        Returns the complete text of a .xml file, closing the document after
        we are done reading it and yielding it in a memory-safe fashion.
        """

        # Create a generator, loading one document into memory at a time.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            encoding = self._get_xml_encoding_from_file(path)
            with codecs.open(path, "r", encoding=encoding) as f:
                doc = f.read()
                if doc:
                    x = etree.fromstring(
                        bytes(doc, encoding="utf-8"),
                        parser=etree.XMLParser(huge_tree=True),
                    )
                    yield etree.tostring(x, pretty_print=True, encoding=str)

    def _format_text(self, text):
        text = re.sub(r"\n{3,}", "\n", text)
        text = "\n".join([item.strip() for item in text.split("\n")])
        return text

    def _format_para(self, para):
        para = " ".join(para.split())
        para = para.replace("- ", "")  # Fix hyphenation
        para = para.replace("\\", "")  # Fix escaping
        return para

    def paras(self, fileids: Union[str, list] = None):
        for body in self.bodies(fileids):
            # Check for lines
            if body.xpath(".//tei:l", namespaces=self.ns):
                content = body.xpath(
                    ".//*[self::tei:title or self::tei:l]", namespaces=self.ns
                )
                types = [item.tag.split("}")[1] for item in content]
                texts = [" ".join(item.itertext()).strip() for item in content]
                text_types = zip(texts, types)

                paras = []
                paras_ = []

                for text, text_type in text_types:
                    if text_type == "title":
                        if paras_:
                            paras.append("\n".join(paras_))
                        paras.append(text)
                        paras_ = []
                    else:
                        paras_.append(text)
                paras.append("\n".join(paras_))

                for para in paras:
                    yield para

            else:

                paras = body.xpath(
                    ".//*[self::tei:title or self::tei:p]", namespaces=self.ns
                )
                paras = [self._format_text(" ".join(para.itertext())) for para in paras]

                for para in paras:
                    yield self._format_para(para)
