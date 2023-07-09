# CLTK Readers
A corpus-reader extension for CLTK

Version 0.5.5; tested on Python 3.10.8, CLTK 1.1.5; LatinCy 3.5.3

## Installation
`pip install -e git+https://github.com/diyclassics/cltk_readers.git#egg=cltk_readers`

## Usage
```
>>> from cltkreaders.lat import LatinTesseraeCorpusReader
>>> tess = LatinTesseraeCorpusReader()
```

```
>>> print(tess.fileids())
['ammianus.rerum_gestarum.part.14.tess', 'ammianus.rerum_gestarum.part.15.tess', 'ammianus.rerum_gestarum.part.16.tess', 'ammianus.rerum_gestarum.part.17.tess', ...]
```

```
>>> print(next(tess.tokenized_sents('vergil.aeneid.part.1.tess', simple=True)))
['Arma', 'virumque', 'cano', ',', 'Troiae', 'qui', 'primus', 'ab', 'oris', 'Italiam', ',', 'fato', 'profugus', ',', 'Laviniaque', 'venit', 'litora', ',', 'multum', 'ille', 'et', 'terris', 'iactatus', 'et', 'alto', 'vi', 'superum', 'saevae', 'memorem', 'Iunonis', 'ob', 'iram', ';']
```

```
>>> pprint(next(tess.concordance('vergil.aeneid.part.1.tess')))
{
  ...
  'sensit': [('<verg. aen. 1.125>', 2)],
  'sententia': [('<verg. aen. 1.237>', 4),
               ('<verg. aen. 1.260>', 4),
               ('<verg. aen. 1.582>', 5)],
  'septem': [('<verg. aen. 1.71>', 3),
            ('<verg. aen. 1.170>', 1),
            ('<verg. aen. 1.192>', 4),
            ('<verg. aen. 1.383>', 1)],
  ...
}
```

## Corpora supported (so far!)
- [CLTK Tesserae Latin Corpus](https://github.com/cltk/lat_text_tesserae)
- [CLTK Tesserae Greek Corpus](https://github.com/cltk/grc_text_tesserae)
- [Perseus Humanist and Renaissance Italian Poetry in Latin (PDILL)](https://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:PDILL)
- [Latin Library](https://www.thelatinlibrary.com/)
- [Perseus Dependency Treebanks (AGLDT)](https://perseusdl.github.io/treebank_data/)
- [Universal Dependency treebanks (UD)](https://universaldependencies.org/)
- [Open Greek & Latin CSEL files](https://github.com/OpenGreekAndLatin/csel-dev)
- [CAMENA (jovanovic fork)](https://github.com/nevenjovanovic/camena-neolatinlit)

## Change log
- 0.5.6: Bump spaCy version
- 0.5.5: Update CSEL reader; Update spaCy dependency to LatinCy [lg model](https://huggingface.co/diyclassics/la_core_web_lg)
- 0.5.4: Update spaCy dependency to LatinCy [md model](https://huggingface.co/diyclassics/la_core_web_md)
- 0.5.3: Update spaCy dependency to [md model](https://huggingface.co/diyclassics/la_dep_cltk_md)
- 0.5.2: Minor fixes
- 0.5.1: Fix spaCy model installation
- 0.5.0: Update packaging for PyPI
- 0.4.6: Add `simple` parameter to Tesserae `tokenized_sents`; add `pos_sents` to Tesserae; update demo notebook
- 0.4.5: Update spaCy dependency to [la_dep_cltk_sm-0.2.0](https://github.com/diyclassics/la_dep_cltk_sm)
- 0.4.4: Add support for [Camena](https://github.com/nevenjovanovic/camena-neolatinlit)
- 0.4.3: Add support for Open Greek & Latin [CSEL files](https://github.com/OpenGreekAndLatin/csel-dev)
- 0.4.2: Update lxml; also update spaCy dependency (now to main spaCy project, as of v. 3.4.2)
- 0.4.1: Update spaCy dependency
- 0.4.0: Add support for Latin Library (and similar plaintext collections)
- 0.3.0: Add support for Perseus-style TEI/XML files; add Latin spaCy support for lemmatization and POS tagging
- 0.2.4: Add support for Universal Dependencies files
- 0.2.3: Add support for Perseus AGLDT Treebanks

*Coded 2022-2023 by [Patrick J. Burns](http://github.com/diyclassics)*
