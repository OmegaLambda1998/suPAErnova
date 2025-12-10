from typing import TYPE_CHECKING

from supaernova._tf import tf

from .utils import pf


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


def pp(expression: object, name: str | None = None) -> None:
    if isinstance(expression, tf.Tensor):
        db(expression, name or "Tensor", verbose=True, debug=False)
    elif name is not None:
        tf.print(f"{name}:\n{pf(expression)}")
    else:
        tf.print(f"{pf(expression)}")
