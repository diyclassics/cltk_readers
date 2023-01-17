"""

"""

import codecs
import os
import tarfile
from typing import Union, Callable, Iterator

from cltk.utils import get_cltk_data_dir
from lxml import etree
from requests import get

from cltkreaders.formats.cora import CORAReader
from cltkreaders.readers import XMLCorpusReader
from cltkreaders.utils import create_default_license_file


class ReferenzkorpusMhdReader(XMLCorpusReader):

    def __init__(self, root, fileids, wrap_etree=False):

        self._wrap_etree = wrap_etree

        self.filename = "rem-corralled-20161222.tar.xz"
        self.lang = "mhg"
        self.corpus = "rem"
        self.corpus_path = os.path.join(get_cltk_data_dir(), f"{self.lang}", "text", f"{self.corpus}", "texts")

        create_default_license_file(self.corpus_path)

        self._download()

        super().__init__(root, fileids, wrap_etree)

    def _download(self):
        """
        >>> rem_reader = ReferenzkorpusMhdReader(os.path.join(get_cltk_data_dir(), "mhg", "text", "rem", "texts"), '.*.xml')
        >>> rem_reader._download()

        """
        full_path = os.path.join(self.corpus_path, self.filename)

        if not os.path.exists(self.corpus_path):
            os.makedirs(self.corpus_path)

        if not os.path.exists(full_path):
            with get("https://zenodo.org/record/3624693/files/rem-corralled-20161222.tar.xz?download=1", stream=True) as g:
                g.raise_for_status()
                with open(full_path, "wb") as f:
                    for chunk in g.iter_content(chunk_size=8192):
                        f.write(chunk)
            with tarfile.open(full_path) as f:
                f.extractall(self.corpus_path)
            corpus_directory = os.path.join(self.corpus_path, "rem-corralled-20161222")
            for filename in os.listdir(corpus_directory):
                os.rename(os.path.join(corpus_directory, filename), os.path.join(self.corpus_path, filename))
            # if os.path.exists(corpus_directory):
            #     os.remove(corpus_directory)

    def docs(self,  fileids: Union[str, list] = None):
        """
        >>> em_reader = ReferenzkorpusMhdReader(os.path.join(get_cltk_data_dir(), "mhg", "text", "rem", "texts"), '.*.xml')
        >>> em_reader.corpus
        'rem'
        >>> em_reader.license()
        ''
        >>> file_ids = em_reader.fileids()
        >>> file_ids[0]
        'M001-N1.xml'

        # >>> doc = em_reader.docs('M001-N1.xml')
        # >>> docs = [i for i in doc]
        #
        # # >>> print(docs[0])
        #
        # >>> doc1 = docs[0]
        # >>> tree = em_reader.doc("M321-G1.xml")
        # # >>> root = tree.tree.getroot()
        # # >>> help(root)
        # # >>> root.items()
        # # [('id', 'M321-G1')]
        # # >>> root.getchildren()
        # #
        # # >>> tree.cora_headers
        # #
        # #
        # # >>> tree.layout_info
        # #
        # # >>> tree.shift_tags
        # #
        # # >>> t = tree.tokens[0]
        # # >>> node = t.tok_anno.node
        # # >>> node.getchildren()
        # #
        # # >>> [n.tag for n in node.getchildren()]
        # #
        # # >>> {n.tag: n.items() for n in node.getchildren()}
        # #
        # # >>> t.tok_anno.to_dict()
        # #
        # >>> [w for w in em_reader.words("M321-G1.xml")][:15]

        # >>> [w for w in em_reader.sents("M321-G1.xml")][:10]

        # >>> [w for w in em_reader.sents("M321-G1.xml", normalized=False)][:10]

        >>> for i, l in enumerate(em_reader.lines("M321-G1.xml", normalized=False)):
        ...    print([w.tok_anno for w in l])
        ...    if i > 10:
        ...        break


        Returns the complete text of a .xml file, closing the document after
        we are done reading it and yielding it in a memory-safe fashion.
        """

        # Create a generator, loading one document into memory at a time.
        for tree in self.docs_tree(fileids):
            yield etree.tostring(tree, pretty_print=True, encoding=str)

    def docs_tree(self, fileids):
        parser = etree.XMLParser(huge_tree=True)
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                # doc = f.read()
                # if doc:
                yield etree.parse(f, parser=parser)
                # yield etree.fromstring(bytes(doc, encoding='utf-8'), parser=parser)

    def doc(self, file_id: str):
        for path, encoding in self.abspaths(file_id, include_encoding=True):
            tree = etree.parse(path, parser=etree.XMLParser(huge_tree=True))
            return CORAReader(tree)
        return None
        # for i in self.docs_tree(file_id):
        #     return i

        # parser = etree.XMLParser(huge_tree=True)
        # tree = etree.parse(os.path.join(file_id), parser=parser)
        # return tree.getroot()

    def words(self, fileids: Union[list, str] = None, preprocess: Callable = None, normalized=True) -> Iterator[str]:
        for doc in self.docs_tree(fileids):
            for word in CORAReader(doc).words(normalized=normalized):
                if preprocess:
                    yield preprocess(word)
                else:
                    yield word

    def sents(self, fileids: Union[list, str] = None, normalized=True) -> Iterator[str]:
        for doc in self.docs_tree(fileids):
            for sent in CORAReader(doc).sents(normalized=normalized):
                yield sent

    def lines(self, fileids: Union[list, str] = None, normalized=True) -> Iterator[str]:
        for doc in self.docs_tree(fileids):
            for line in CORAReader(doc).lines():
                yield line


