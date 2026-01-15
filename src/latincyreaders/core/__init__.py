"""Core abstractions for latincy-readers."""

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel
from latincyreaders.core.protocols import CorpusReaderProtocol

__all__ = ["BaseCorpusReader", "AnnotationLevel", "CorpusReaderProtocol"]
