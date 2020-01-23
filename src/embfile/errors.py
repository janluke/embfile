class Error(Exception):
    """ Base class of all errors raised by embfile """


class IllegalOperation(Error):
    """ Raised when the user attempts to perform an operation that is illegal in the current state
    (e.g. using a closed file) """


class BadEmbFile(Error):
    """ Raised when the file is malformed. """
