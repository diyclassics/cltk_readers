
import os
import tarfile


from cltk.utils import get_cltk_data_dir
from requests import get

from cltkreaders.readers import XMLCorpusReader


class ReferenzkorpusMhdReader(XMLCorpusReader):

    def __init__(self, root, fileids, wrap_etree=False):
        self._wrap_etree = wrap_etree
        XMLCorpusReader.__init__(self, root, fileids)

        self.lang = "mhg"
        self.corpus = "rem"

    def download(self):
        """
        >>> rem_reader = ReferenzkorpusMhdReader()
        >>> rem_reader.download()

        """
        filename = "rem-corralled-20161222.tar.xz"
        corpus_path = os.path.join(
            get_cltk_data_dir(),
            f"{self.lang}", "text", f"{self.corpus}", "texts")
        if not os.path.exists(corpus_path):
            os.makedirs(corpus_path)
        with get("https://zenodo.org/record/3624693/files/rem-corralled-20161222.tar.xz?download=1", stream=True) as g:
            g.raise_for_status()
            with open(os.path.join(corpus_path, filename), "wb") as f:
                for chunk in g.iter_content(chunk_size=8192):
                    f.write(chunk)
        with tarfile.open(os.path.join(corpus_path, filename)) as f:
            f.extractall(corpus_path)


    def