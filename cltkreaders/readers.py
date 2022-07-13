"""Corpus readers to support text collections for use with CLTK;
    Code following work in Bengfort, B., Bilbro, R., and Ojeda, T. 2018. Applied Text Analysis with Python:
    Enabling Language-Aware Data Products with Machine Learning. Sebastopol, CA: O’Reilly.
"""

__author__ = ["Patrick J. Burns <patrick@diyclassics.org>",]
__license__ = "MIT License."

import os
import os.path
import glob
import json
from collections import defaultdict 
import warnings
import codecs
import unicodedata
import time
import re
from typing import Callable, DefaultDict, Iterator, Union

import xml.etree.ElementTree as ET
from lxml import etree

from nltk import FreqDist
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.xmldocs import XMLCorpusReader

from cltk.sentence.lat import LatinPunktSentenceTokenizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer
from cltk.sentence.grc import GreekRegexSentenceTokenizer
from cltk.tokenizers.word import PunktWordTokenizer as GreekWordTokenizer

from pyuca import Collator
c = Collator()

class CLTKCorpusReaderMixin():
    def load_metadata(self):
        jsonfiles = glob.glob(f'{self.root}/**/metadata/*.json', recursive=True)
        jsons = [json.load(open(file)) for file in jsonfiles]
        merged = defaultdict(dict)
        for json_ in jsons:
            for k, v in json_.items():
                merged[k].update(v)
        return merged

    def metadata(self, label: str, fileids: Union[str, list] = None):
        if not label:
            return None
        # if not fileids:
        #     fileids = self.fileids()

        if isinstance(fileids, np.ndarray) and not fileids.size:
            fileids = self.fileids()
        elif isinstance(fileids, (list)) and not fileids:
            fileids = self.fileids()            

        #TODO: Shouldn't self.fileids() handle str/lst    
        if isinstance(fileids, str):
            record = self._metadata.get(fileids, None)
            if record:
                return record.get(label, None)
            else:
                return None
        else:
            records = [self._metadata.get(fileid, None) for fileid in fileids]
            label_records = [record.get(label, None) if record else None for record in records]
            return label_records     
    
    def sizes(self, fileids: Union[list, str] = None) -> Iterator[int]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :yield: Size of UD document
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

    def citation(self):
        citations = glob.glob(f'{self.root}/**/citation.bib', recursive=True)
        # citations += glob.glob(f'{self.root}/**/CITATION.bib', recursive=True)

        citation_full = []

        for citation in citations:
            citation_header = f'Citation for files in folder {os.path.dirname(citation)}'
            with open(citation, 'r') as f:
                citation_text = f.read()
                citation_full.append(f'{citation_header}\n\n{citation_text}')
        
        return '\n\n'.join(citation_full)

    def license(self):
        licenses = [file for file in glob.glob(f'{self.root}/**/*.*',recursive=True) if 'license.txt' in file.lower() or 'license.md' in file.lower()] 
        
        license_full = []

        for license in licenses:
            license_header = f'License for files in folder {os.path.dirname(license)}'
            with open(license, 'r') as f:
                license_text = f.read()
                license_full.append(f'{license_header}\n\n{license_text}')
        
        return '\n\n'.join(license_full)


class TesseraeCorpusReader(CLTKCorpusReaderMixin, PlaintextCorpusReader):
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
                

class TEICorpusReader(XMLCorpusReader):
    """
    A corpus reader for working TEI/XML docs
    """

    def __init__(self, root, fileids: str = r'.*\.xml', encoding='utf8', **kwargs):
        """
        """
        XMLCorpusReader.__init__(self, root, fileids)
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def docs(self,  fileids: Union[str, list] = None):
        """
        Returns the complete text of a .xml file, closing the document after
        we are done reading it and yielding it in a memory-safe fashion.
        """

        # Create a generator, loading one document into memory at a time.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                doc = f.read()
                if doc:
                    x = etree.fromstring(bytes(doc, encoding='utf-8'), parser=etree.XMLParser(huge_tree=True))
                    yield etree.tostring(x, pretty_print=True, encoding=str)

    def bodies(self, fileids: Union[str, list] = None):
        for doc in self.docs(fileids):
            root = ET.fromstring(doc)
            body = root.find(f'.//tei:body', self.ns)
            yield body


class PerseusTreebankCorpusReader(TEICorpusReader):
    """
    A corpus reader for working with Perseus Treebank (AGLDT) files, v.2.1; files can be found
    here: https://github.com/PerseusDL/treebank_data/tree/master/v2.1

    NB: `root` should point to a directory containing the AGLDT files
    """

    def __init__(self, root: str, fileids: str = r'.*\.xml', encoding: str = 'utf8', **kwargs):
        TEICorpusReader.__init__(self, root, fileids, encoding=encoding)

    def bodies(self, fileids: Union[str, list] = None):
        for doc in self.docs(fileids):
            root = ET.fromstring(doc)
            body = root.find(f'.//body')
            yield body

    def paras(self, fileids: Union[str, list] = None):
        for body in self.bodies(fileids):
            paras = body.findall('.//p')
            # If no paras available, return entire body as a 'para'
            if not paras:
                paras = [body]
            for para in paras:
                yield para           

    def sents(self, fileids: Union[str, list] = None, plaintext: bool = False):
        for para in self.paras(fileids):
            sents = para.findall('.//sentence')
            for sent in sents:
                if plaintext:
                    sent = ' '.join([word.get('form', '') for word in sent.findall('.//word')])
                    sent = re.sub(r'\[\d+\]', '', sent) # Clean up 'insertion' tokens
                yield sent

    def word_data(self, fileids: Union[str, list] = None):
        for sent in self.sents(fileids):
            words = sent.findall('.//word')
            for word in words:
                yield word

    def words(self, fileids: Union[str, list] = None):
        for word in self.word_data(fileids):
            yield word.get('form', None)
    
    def tokenized_sents(self, fileids=None, simple_pos: bool = True):
        for para in self.paras(fileids):
            sents = para.findall('.//sentence')
            for sent in sents:
                tokenized_sent = []
                words = sent.findall('.//word')
                for word in words:
                    token = word.get('form', None)
                    lemma = word.get('lemma', None)
                    postag = word.get('postag', None)
                    if simple_pos and postag:
                        postag = postag[0].upper() # TODO: Write tag map?
                    tokenized_sent.append((token, lemma, postag))
                yield tokenized_sent                


class UDCorpusReader(CLTKCorpusReaderMixin, CorpusReader):
    """
    Generic corpus reader for texts from the UD treebanks
    """
    def __init__(self, root: str, fileids: str = r'.*\.conllu', 
                 columns: list = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'], 
                 encoding: str = 'utf-8', lang: str = None, normalization_form: str = 'NFC', **kwargs):
        """
        :param root: Location of conllu files to be read into corpus reader
        :param fileids: Pattern for matching files to be read into corpus reader
        :param encoding: Text encoding for associated files; defaults to 'utf-8'
        :param lang: Allows a language to be selected for language-specific corpus tasks
        :param normalization_form: Normalization form for associated files; defaults to 'NFC'
        :param kwargs: Miscellaneous keyword arguments
        """
        if lang:
            self.lang = lang.lower()
        self.columns = columns
        self.normalization_form = normalization_form
        CorpusReader.__init__(self, root, fileids, encoding)

    def docs(self, fileids: Union[list, str] = None) -> Iterator[str]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :yield: Plaintext content of UD file
        """
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                doc = f.read()
                doc = unicodedata.normalize(self.normalization_form, doc)
                doc = doc.strip()
                yield doc
    
    texts = docs # alias for docs

    def paras(self, fileids: Union[list, str] = None):
        """
        UD documents do not have include paragraph divisions
        """
        raise NotImplementedError

    def sent_blocks(self, fileids: Union[list, str] = None) -> Iterator[str]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :return: Sentence-level blocks of text in UD files 
        """
        for doc in self.docs(fileids):
            sent_blocks_ = doc.split('\n\n')
            for sent_block in sent_blocks_:
                yield sent_block

    def sent_dicts(self, fileids: Union[list, str] = None) -> Iterator[str]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :return: Sentence-level blocks divided as dictionaries with columnar data 
        """        
        for sent_block in self.sent_blocks(fileids):
            sent_lines = sent_block.split('\n')
            data_lines = [line.split('\t') for line in sent_lines if not line.startswith('#') and line]
            sent_dicts_ = []
            for data_line in data_lines:
                data_line_ = {k: v for k, v in zip(self.columns, data_line)}
                sent_dicts_.append(data_line_)
            yield sent_dicts_

    def sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: List of sentences from Tesserae documents
        """        
        for sent_block in self.sent_blocks(fileids):
            for row in sent_block.split('\n'):
                if row.startswith('# text = '):
                    sent = row.replace('# text = ', '')
                    if preprocess:
                        yield preprocess(sent)
                    else:
                        yield sent

    def tokenized_sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: List of words from UD documents
        """
        for sent in self.sent_dicts(fileids):
            tokenized_sent = [item['FORM'] for item in sent]
            if preprocess:
                yield preprocess(" ".join(tokenized_sent)).split()
            else:
                yield tokenized_sent

    def lemmatized_sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: List of words from UD documents
        """
        for sent in self.sent_dicts(fileids):
            tokenized_sent = [item['LEMMA'] for item in sent]
            if preprocess:
                yield preprocess(" ".join(tokenized_sent)).split()
            else:
                yield tokenized_sent                

    def words(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: List of words from UD documents
        """
        for tokenized_sent in self.tokenized_sents(fileids, preprocess=preprocess):
            for token in tokenized_sent:
                yield token

    def lemmas(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: List of words from UD documents
        """
        for lemmatized_sent in self.lemmatized_sents(fileids, preprocess=preprocess):
            for lemma in lemmatized_sent:
                yield lemma

    def pos_sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        :param fileids: Subset of files to be processed by reader tasks
        :param preprocess: Allows a preprocessing function to be passed to reader task
        :yield: List of data from UD documents in form TOKEN/POS
        """
        for sent in self.sent_dicts(fileids):
            # pos_sent = [f'{item["FORM"]}/{item["UPOS"]}' for item in sent]
            tokenized_sent = [item['FORM'] for item in sent]
            pos_sent = [item['UPOS'] for item in sent]
            if preprocess:
                tokenized_sent = [preprocess(token).replace(' ','') for token in tokenized_sent]
            pos_sent = zip(tokenized_sent, pos_sent)
            pos_sent = [f"{item[0]}/{item[1]}" for item in pos_sent if item[0]]
            yield pos_sent

    def annotated_sents(self, fileids: Union[list, str] = None, preprocess: Callable = None) -> Iterator[list]:
        """
        """
        for sent in self.sent_dicts(fileids):
            token_sent = [item['FORM'] for item in sent]
            lemma_sent = [item['LEMMA'] for item in sent]
            pos_sent = [item['UPOS'] for item in sent]
            
            if preprocess:
                token_sent = [preprocess(token) for token in token_sent]
                lemma_sent = [preprocess(lemma) for lemma in lemma_sent]

            annotated_sent = list(zip(token_sent, lemma_sent, pos_sent))
            yield annotated_sent
