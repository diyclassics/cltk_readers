# CLTK Readers
A corpus-reader extension for CLTK

Version 0.2.0.; tested on Python 3.9.10, CLTK 1.0.22

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

*Coded 2.9.2022 by [Patrick J. Burns](http://github.com/diyclassics)*
