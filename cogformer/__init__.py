"""CogFormer.

Importing the package applies the project's figure typography (see
``cogformer.utils.style``). It is done here because it is the only import site
every figure-producing script in the repo reliably passes through.
"""
from .utils import style as _style  # noqa: F401  (import for side effect)
