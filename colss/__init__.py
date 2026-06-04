import numpy as np
from colss._colss import query as _query

def query(expr: str, **kwargs) -> np.ndarray:
    arrays = {}
    scalars = {}
    shape = None

    for name, val in kwargs.items():
        if isinstance(val, np.ndarray):
            arr = np.ascontiguousarray(val, dtype=np.float64)
            arrays[name] = arr
            if shape is None:
                shape = val.shape
            elif val.shape != shape:
                raise ValueError(f"Array shape mismatch: {val.shape} vs {shape}")
        elif isinstance(val, (int, float)):
            scalars[name] = float(val)
        else:
            raise TypeError(f"Unsupported variable type for {name}: {type(val)}")

    # Evaluate using the C++ backend
    res = _query(expr, scalars, **arrays)

    # Reshape to the original N-dimensional shape
    if shape is not None:
        return res.reshape(shape)
    return res
