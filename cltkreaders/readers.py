"""Corpus readers to support text collections for use with CLTK;
    Code following work in Bengfort, B., Bilbro, R., and Ojeda, T. 2018. Applied Text Analysis with Python:
    Enabling Language-Aware Data Products with Machine Learning. Sebastopol, CA: O’Reilly.
"""

__author__ = ["Patrick J. Burns <patrick@diyclassics.org>",]
__license__ = "MIT License."

import os
import glob
import json
from collections import defaultdict 
import warnings
import codecs
import unicodedata
import time
import re
from typing import Callable, DefaultDict, Iterator, Union
from collections import defaultdict

from nltk import FreqDist
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

from cltk.sentence.lat import LatinPunktSentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer
from cltk.sentence.grc import GreekRegexSentenceTokenizer
from cltk.tokenizers.word import PunktWordTokenizer as GreekWordTokenizer

from pyuca import Collator
c = Collator()


class TesseraeCorpusReader(PlaintextCorpusReader):
    """
    Generic corpus reader for texts from the Tesserae-CLTK corpus
    """

    def __init__(self, root: str, fileids: str = None, encoding: str = 'utf-8', lang: str = None,
                 normalization_form: str = 'NFC',
                 word_tokenizer: Callable = None, sent_tokenizer: Callable = None, **kwargs):
        """
        :param root: Location of plaintext files to be read into corpus reader
        :param fileids: Pattern for matching files to be read into corpus reader
        :param encoding: Text encoding for associated files; defaults to 'utf-8'
        :param lang: Allows a language to be selected for language-specific corpus tasks
        :param normalization_form: Normalization form for associated files; defaults to 'NFC'
        :param kwargs: Miscellaneous keyword arguments
        """

        # Tesserae readers at present are limited to support for Greek and Latin; check on instantiation
        if lang:
            self.lang = lang.lower()

        if lang is None:
            if 'greek' in root or 'grc' in root:
                warnings.warn("lang parameter inferred from document path and set to 'grc'")
                self.lang = 'grc'
            elif 'lat' in root:
                warnings.warn("lang parameter inferred from document path and set to 'lat'")
                self.lang = 'lat'
            else:
                raise TypeError("lang parameter in TesseraeCorpusReader must be set to 'greek' or 'latin'")
        elif lang == 'greek':
            self.lang = 'grc'
        elif lang == 'latin':
            self.lang = 'lat'
        else:
            if lang not in ['greek', 'grc', 'latin', 'lat']:
                raise TypeError("lang parameter in TesseraeCorpusReader must be set to 'grc' or 'lat'")

        if not sent_tokenizer:
            if self.lang == 'grc':
                self.sent_tokenizer = GreekRegexSentenceTokenizer()
            if self.lang == 'lat':
                self.sent_tokenizer = LatinPunktSentenceTokenizer()

        if not word_tokenizer:
            if self.lang == 'grc':
                self.word_tokenizer = GreekWordTokenizer()
            if self.lang == 'lat':
                self.word_tokenizer = LatinWordTokenizer()

        self.normalization_form = normalization_form
        
        PlaintextCorpusReader.__init__(self, root, fileids, encoding, kwargs)

    @property
    def metadata_(self):
        jsonfiles = glob.glob(f'{self.root}/metadata/*.json')
        jsons = [json.load(open(file)) for file in jsonfiles]
        merged = defaultdict(dict)
        for json_ in jsons:
            for k, v in json_.items():
                merged[k].update(v)
        return merged

    def docs(self, fileids: Union[list, str] = None) -> Iterator[str]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :yield: Plaintext content of Tesserae file
        """
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                doc = f.read()
                doc = unicodedata.normalize(self.normalization_form, doc)
                doc = doc.strip()
                yield doc

    def paras(self, fileids: Union[list, str] = None):
        """
        Tesserae documents (at present) are not marked up to include paragraph divisions
        """
        raise NotImplementedError

    def doc_rows(self, fileids: Union[list, str] = None) -> Iterator[dict]:
        """
        Provides a convenient dict-based data structure for keeping track of citation information and corresponding
        text data; format is:
        ```{'<lucr. 2.1>': 'Suave, mari magno turbantibus aequora ventis',
            '<lucr. 2.2>': 'e terra magnum alterius spectare laborem;', ...
            }```

        :param fileids: Subset of files to be processed by reader tasks
        :yield: Dictionary with citation as key, text as value
        """
        for doc in self.docs(fileids):
            rows = []

            # Tesserae files are generally divided such that there are no line breaks within citation sections; there
            # are exceptions to protect against, i.e. remove \n when not followed by the citation indicator (<)
            doc = re.sub(r'\n([^<])', r' \g<1>', doc)

            lines = [line for line in doc.split('\n') if line]
            for line in lines:
                try:
                    k, v = line.split('>', 1)
                    k = f'{k}>'
                    v = v.strip()
                    rows.append((k, v))
                except:
                    print(f'The following line is not formatted corrected and has been skipped: {line}\n')
            yield dict(rows)

    def texts(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[str]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: Plaintext content of Tesserae file with citation information removed
        """
        for doc_row in self.doc_rows(fileids):
            text = '\n'.join(doc_row.values())
            if preprocess:
                text = preprocess(text)
            yield text

    def sents(self, fileids: Union[list, str] = None, preprocess: Callable = None,
              unline: bool = False) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :param unline: If `True`, removes newline characters from returned sents; useful in normalizing verse
        :yield: List of sentences from Tesserae documents
        """
        for text in self.texts(fileids):
            sents = self.sent_tokenizer.tokenize(text)
            if unline:
                sents = [sent.replace('\n', ' ') for sent in sents]
            for sent in sents:
                if preprocess:
                    yield preprocess(sent)
                else:
                    yield sent

    def tokenized_sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: List of words from Tesserae documents
        """
        for sent in self.sents(fileids, preprocess=preprocess):
            tokens = self.word_tokenizer.tokenize(sent)
            yield tokens

    def words(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: List of words from Tesserae documents
        """
        for tokenized_sent in self.tokenized_sents(fileids, preprocess=preprocess):
            for token in tokenized_sent:
                yield token

    def pos_sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        # TODO: add pos_sent method (cf. Bengfort et al. pp. 44-45)
        # TODO: check CLTK Latin pos tagger
        raise NotImplementedError

    def concordance(self, fileids: Union[list, str] = None, preprocess: Callable = None, compiled=False)\
            -> Iterator[dict]:
        """
        Provides a concordance-style data structure, i.e. dictionary with word as key and list of citation/locations
        as value
        text data; format is:
        ```{'a': [('<il. lat. 103>', 7),
                   ('<il. lat. 480>', 3),
                   ('<il. lat. 619>', 2),
                   ('<il. lat. 1044>', 1)],
             'ab': [('<il. lat. 35>', 5),
                    ('<il. lat. 90>', 1), etc...```
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :param compiled: If `True`, a single compiled concordance is created for each file in fileids; if `False`,
            a generator with a concordance for each file is created.
        :yield: Dictionary with token as key, list of citations as value
        """

        def build_concordance(doc_rows):
            concordance_dict = defaultdict(list)
            items = doc_rows.items()
            for citation, text in items:
                if preprocess:
                    text = preprocess(text)
                text_tokens = text.split()
                for i, token in enumerate(text_tokens):
                    concordance_dict[token].append((citation, i))
            return sorted(concordance_dict.items(), key=lambda x: c.sort_key(x[0]))
        
        if compiled:
            concordance_dict_compiled = defaultdict(list)
            for doc_rows in self.doc_rows(fileids):
                conc = build_concordance(doc_rows)
                for token, refs in conc:
                    concordance_dict_compiled[token].extend(refs)
            yield dict(sorted(concordance_dict_compiled.items(), key=lambda x: c.sort_key(x[0])))
        else:
            for doc_rows in self.doc_rows(fileids):
                yield dict(build_concordance(doc_rows))

    def citation(self):
        with open(f'{self.root}/../citation.bib', 'r') as f:
            return f.read()   

    def license(self):
        with open(f'{self.root}/../LICENSE.md', 'r') as f:
            return f.read()

    def metadata(self, label, fileids=None):
        if not label:
            return None
        if not fileids:
            fileids = self.fileids()

        #TODO: Shouldn't self.fileids() handle str/lst    
        if isinstance(fileids, str):
            record = self.metadata_.get(fileids, None)
            if record:
                return record.get(label, None)
            else:
                return None
        else:
            records = [self.metadata_.get(fileid, None) for fileid in fileids]
            label_records = [record.get(label, None) if record else None for record in records]
            return label_records  

    def sizes(self, fileids: Union[list, str] = None) -> Iterator[int]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :yield: Size of Tesserae document
        """
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

    def describe(self, fileids: Union[list, str] = None) -> dict:
        """
        Performs a single pass of the corpus and returns a dictionary with a variety of metrics concerning the state
        of the corpus.
        """
        started = time.time()

        # Structures to perform counting.
        counts = FreqDist()
        tokens = FreqDist()

        # Perform single pass over paragraphs, tokenize and count
        for sent in self.sents(fileids):
            counts['sents'] += 1

        for word in self.words(fileids):
            counts['words'] += 1
            tokens[word] += 1

        # Compute the number of files and categories in the corpus
        if isinstance(fileids, str):
            n_fileids = 1
        elif isinstance(fileids, list):
            n_fileids = len(fileids)
        else:
            n_fileids = len(self.fileids())

        # Return data structure with information
        return {
            'files': n_fileids,
            'sents': counts['sents'],
            'words': counts['words'],
            'vocab': len(tokens),
            'lexdiv': float(counts['words']) / float(len(tokens)),
            'secs': time.time() - started,
        }
