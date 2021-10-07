

# # Code based on Bengfort et al. 2018
#
# import codecs
# import time
#
# from nltk import FreqDist
# from nltk.corpus.reader.api import CorpusReader
#
#
# class UDCorpusReader(CorpusReader):
#     """
#     A corpus reader for texts from the UD treebanks
#     """
#
#     def __init__(self, root, fileids=None, columns=None, encoding='utf-8', **kwargs):
#         """
#         Write doc string
#         :param root:
#         :param fileids:
#         :param encoding:
#         :param kwargs:
#         """
#
#         CorpusReader.__init__(self, root, fileids, encoding)
#         self.columns = columns
#
#     def docs(self, fileids=None):
#         """
#         :param fileids:
#         :return:
#         """
#         for path, encoding in self.abspaths(fileids, include_encoding=True):
#             with codecs.open(path, 'r', encoding=encoding) as f:
#                 doc = f.read()
#                 yield doc
#
#     def paras(self, fileids=None):
#         raise NotImplementedError
#
#     def texts(self, fileids=None):
#         raise NotImplementedError
#
#     def sent_blocks(self, fileids=None):
#         """
#         :param fileids:
#         :return:
#         """
#         for doc in self.docs(fileids):
#             sent_blocks_ = doc.split('\n\n')
#             for sent_block in sent_blocks_:
#                 yield sent_block
#
#     def sent_dicts(self, fileids=None):
#         for sent_block in self.sent_blocks(fileids):
#             sent_lines = sent_block.split('\n')
#             data_lines = [line.split('\t') for line in sent_lines if not line.startswith('#') and line]
#             sent_dicts_ = []
#             for data_line in data_lines:
#                 data_line_ = {k: v for k, v in zip(self.columns, data_line)}
#                 sent_dicts_.append(data_line_)
#             yield sent_dicts_
#
#     def sents(self, fileids=None):
#         for sent_block in self.sent_blocks(fileids):
#             for row in sent_block.split('\n'):
#                 if row.startswith('# text = '):
#                     yield row.replace('# text = ', '')
#
#     def words(self, fileids=None):
#         """
#         :param fileids:
#         :return:
#         """
#         for sent_dict in self.sent_dicts(fileids):
#             for item in sent_dict:
#                 yield item['FORM']
#
#     def lemmas(self, fileids=None):
#         """
#         :param fileids:
#         :return:
#         """
#         for sent_dict in self.sent_dicts(fileids):
#             for item in sent_dict:
#                 yield item['LEMMA']
#
#     def tagged(self, fileids=None):
#         for sent_dict in self.sent_dicts(fileids):
#             for item in sent_dict:
#                 yield item['FORM'], item['UPOS']
#
#     def describe(self, fileids=None):
#         """
#         Performs a single pass of the corpus and returns a dictionary with a variety of metrics concerning the state
#         of the corpus.
#         """
#         started = time.time()
#
#         # Structures to perform counting.
#         counts = FreqDist()
#         tokens = FreqDist()
#
#         # Perform single pass over paragraphs, tokenize and count
#         for _ in self.sents(fileids):
#             counts['sents'] += 1
#
#         for word in self.words(fileids):
#             counts['words'] += 1
#             tokens[word] += 1
#
#         # Compute the number of files and categories in the corpus
#         if isinstance(fileids, str):
#             n_fileids = 1
#         elif isinstance(fileids, list):
#             n_fileids = len(fileids)
#         else:
#             n_fileids = len(self.fileids())
#
#         # Return data structure with information
#         return {
#             'files': n_fileids,
#             'sents': counts['sents'],
#             'words': counts['words'],
#             'vocab': len(tokens),
#             'lexdiv': float(counts['words']) / float(len(tokens)),
#             'secs': time.time() - started,
#         }
#
#
# if __name__ == '__main__':
#     root = '~/workspace/repos/UD_Latin-ITTB'  # TODO change to CLTK data location
#     DOC_PATTERN = '.*\.conllu'
#     columns = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
#     reader = UDCorpusReader(root, DOC_PATTERN, columns)
#     print(reader.describe())
