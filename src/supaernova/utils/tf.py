from typing import TYPE_CHECKING

from supaernova._tf import tf

from .utils import pf


def pp(expression: object) -> None:
    tf.print(pf(expression))


def db(
    tensor: "tf.Tensor",
    name: str,
    *,
    dtype: bool = True,
    shape: bool = True,
    value: bool = True,
    verbose: bool = False,
    debug: bool = True,
) -> None:
    msg = name
    if dtype:
        msg += f": {tensor.dtype}"
    if shape:
        msg += f": {tensor.shape}"
    if value:
        msg += "\n" + pf(tensor)

    if verbose:
        print(msg)
    if debug:
        __import__("tensorflow").debugging.check_numerics(tensor, message=msg)
