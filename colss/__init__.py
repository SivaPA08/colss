import inspect
import numpy as np
from colss._colss import sum as _sum
from colss._colss import prod as _prod
from colss._colss import mean as _mean
from colss._colss import query as _query
from colss._colss import sd as _sd
from colss._colss import var as _var
from colss._colss import median as _median

def _collect_vars(expr: str):
    frame = inspect.currentframe()
    frame = frame.f_back.f_back if frame and frame.f_back else None
    arrays = {}
    scalars = {}
    while frame is not None:
        for name, val in frame.f_locals.items():
            if name not in expr:
                continue
            if isinstance(val, np.ndarray):
                if name not in arrays:
                    arrays[name] = np.ascontiguousarray(val, dtype=np.float64)
            elif isinstance(val, (int, float)) and name not in scalars:
                scalars[name] = float(val)
        if arrays or scalars:
            break
        frame = frame.f_back
    if not arrays and not scalars:
        raise RuntimeError(f"No variables found matching '{expr}'")
    return arrays, scalars

def sum(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _sum(expr, scalars, **arrays)

def prod(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _prod(expr, scalars, **arrays)

def mean(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _mean(expr, scalars, **arrays)

def query(expr: str) -> np.ndarray:
    arrays, scalars = _collect_vars(expr)
    return _query(expr, scalars, **arrays)

def sd(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _sd(expr, scalars, **arrays)

def var(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _var(expr, scalars, **arrays)

def median(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _median(expr, scalars, **arrays)
