__all__ = [
    "VLinException",
    "IntegerNotSupported",
]


class VLinException(Exception):
    """ Base class for all exceptions coming from the vlin library. """


class IntegerNotSupported(Exception):
    """ Specified solver does not support integer variables. """
