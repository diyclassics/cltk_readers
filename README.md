# CLTK Readers
A corpus-reader extension for CLTK

Version 0.4.2; tested on Python 3.10.4, CLTK 1.1.5

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
>>> print(next(tess.tokenized_sents('vergil.aeneid.part.1.tess')))
['Arma', 'virum', '-que', 'cano', ',', 'Troiae', 'qui', 'primus', 'ab', 'oris', 'Italiam', ',', 'fato', 'profugus', ',', 'Lavinia', '-que', 'venit', 'litora', ',', 'multum', 'ille', 'et', 'terris', 'iactatus', 'et', 'alto', 'vi', 'superum', 'saevae', 'memorem', 'Iunonis', 'ob', 'iram', ';', 'multa', 'quoque', 'et', 'bello', 'passus', ',', 'dum', 'conderet', 'urbem', ',', 'inferret', '-que', 'deos', 'Latio', ',', 'genus', 'unde', 'Latinum', ',', 'Albani', '-que', 'patres', ',', 'atque', 'altae', 'moenia', 'Romae', '.']
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

## Change log
- 0.4.2: Update lxml; also update spaCy dependency (now to main spaCy project, as of v. 3.4.2)
- 0.4.1: Update spaCy dependency
- 0.4.0: Add support for Latin Library (and similar plaintext collections)
- 0.3.0: Add support for Perseus-style TEI/XML files; add Latin spaCy support for lemmatization and POS tagging
- 0.2.4: Add support for Universal Dependencies files
- 0.2.3: Add support for Perseus AGLDT Treebanks

*Coded 2022 by [Patrick J. Burns](http://github.com/diyclassics)*
