import re
import numpy as np
from colss._colss import query as _query

def inline_scalars(expr: str, scalars: dict[str, float]) -> str:
    for name, value in sorted(scalars.items(), key=lambda kv: -len(kv[0])):
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
        expr = re.sub(pattern, f"({float(value)!r})", expr)
    return expr

def query(expr: str, **kwargs) -> np.ndarray:
    arrays = {}
    scalars = {}
    shape = None

    for name, val in kwargs.items():
        if isinstance(val, np.ndarray):
            arr = np.asarray(val, dtype=np.float64, order="C")
            arrays[name] = arr

            if shape is None:
                shape = arr.shape
            elif arr.shape != shape:
                raise ValueError(
                    f"Array shape mismatch for '{name}': {arr.shape} vs {shape}"
                )

        elif isinstance(val, (int, float, np.integer, np.floating)):
            scalars[name] = float(val)

        else:
            raise TypeError(
                f"Unsupported variable type for '{name}': {type(val).__name__}"
            )

    expr = inline_scalars(expr, scalars)
    res = _query(expr, **arrays)

    if shape is not None and res.shape != shape:
        res = res.reshape(shape)

    return res
