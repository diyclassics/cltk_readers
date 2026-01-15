"""
latincy-readers: Corpus readers for Latin texts with LatinCy/spaCy integration.

Example usage:
    from latincyreaders import TesseraeReader, PlaintextReader, LatinLibraryReader
    from latincyreaders import TEIReader, PerseusReader

    reader = TesseraeReader("/path/to/tesserae")
    for doc in reader.docs():
        print(doc.text[:100])
"""

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.readers.tesserae import TesseraeReader
from latincyreaders.readers.plaintext import PlaintextReader, LatinLibraryReader
from latincyreaders.readers.tei import TEIReader, PerseusReader

__version__ = "1.0.0a1"
__all__ = [
    "TesseraeReader",
    "PlaintextReader",
    "LatinLibraryReader",
    "TEIReader",
    "PerseusReader",
    "AnnotationLevel",
]
