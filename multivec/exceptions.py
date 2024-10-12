"""Various exceptions that can be raised by the library."""


class DOCXLoaderError(Exception):
    """Base exception for DOCXLoader errors."""

    pass


class HTMLLoaderError(Exception):
    """Base exception for HTMLLoader errors."""

    pass


class PDFLoaderError(Exception):
    """Base exception for PDFLoader errors."""

    pass


class LLMAndKeywordsNotFound(Exception):
    """Base exception for LLMAndKeywordsNotFound errors."""

    pass
