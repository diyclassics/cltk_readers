"""
latincy-readers: Corpus readers for Latin texts with LatinCy/spaCy integration.

Example usage:
    from latincyreaders import TesseraeReader

    reader = TesseraeReader()
    for doc in reader.docs():
        print(doc.text[:100])
"""

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.readers.tesserae import TesseraeReader

__version__ = "1.0.0a1"
__all__ = ["TesseraeReader", "AnnotationLevel"]
