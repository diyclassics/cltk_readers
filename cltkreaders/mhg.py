import codecs
import os
import tarfile
from typing import Union

from cltk.utils import get_cltk_data_dir
from lxml import etree
from requests import get

from cltkreaders.readers import XMLCorpusReader


class ReferenzkorpusMhdReader(XMLCorpusReader):

    def __init__(self, root, fileids, wrap_etree=False):

        self._wrap_etree = wrap_etree

        self.filename = "rem-corralled-20161222.tar.xz"
        self.lang = "mhg"
        self.corpus = "rem"
        self.corpus_path = os.path.join(
            get_cltk_data_dir(),
            f"{self.lang}", "text", f"{self.corpus}", "texts")
        if not os.path.exists(os.path.join(self.corpus_path, "LICENSE")):
            with open(os.path.join(self.corpus_path, "LICENSE"), "w") as f:
                f.write("")
        XMLCorpusReader.__init__(self, root, fileids)

    def _download(self):
        """
        >>> rem_reader = ReferenzkorpusMhdReader(os.path.join(get_cltk_data_dir(), "mhg", "text", "rem", "texts"), '.*.xml')
        >>> rem_reader._download()

        """

        if not os.path.exists(self.corpus_path):
            os.makedirs(self.corpus_path)
        if not os.path.exists(os.path.join(self.corpus_path, self.filename)):
            with get("https://zenodo.org/record/3624693/files/rem-corralled-20161222.tar.xz?download=1", stream=True) as g:
                g.raise_for_status()
                with open(os.path.join(self.corpus_path, self.filename), "wb") as f:
                    for chunk in g.iter_content(chunk_size=8192):
                        f.write(chunk)
            with tarfile.open(os.path.join(self.corpus_path, self.filename)) as f:
                f.extractall(self.corpus_path)
        if not os.path.exists(os.path.join(self.corpus_path, "LICENSE")):
            with open(os.path.join(self.corpus_path, "LICENSE"), "r") as f:
                f.write("")

    def docs(self,  fileids: Union[str, list] = None):
        """
        >>> em_reader = ReferenzkorpusMhdReader(os.path.join(get_cltk_data_dir(), "mhg", "text", "rem", "texts"), '.*.xml')
        >>> em_reader.corpus
        'rem'
        >>> em_reader.license()
        ''
        >>> em_reader.fileids()
        >>> doc = em_reader.docs('rem-corralled-20161222/M001-N1.xml')
        >>> print([i for i in doc][0])



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

    def test_get(self):
        """
        >>>
        """
        pass