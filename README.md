# colss
colss is a lightweight expression evaluator that evaluates math-style string expressions on NumPy, Pandas, Polars, and standard Python arrays, reducing verbosity compared to native syntax
eg:
$\Sigma\left(e^{\sin(a)} + \log^2(b+1) - c^3\right)$
```
colss.sum("exp(sin(a)) + log(b+1)^2 - c^3")
```
---
## Installation
```
pip install colss
```
---
## Requirements
All arrays must be 1D, preferably `float64`, and C-contiguous. If you have a 2D array:
```python
a = a.ravel()
```
---
## Usage
All functions accept a string expression. Just pass the expression — no need to register variables or pass arrays manually.
### query
Evaluates an expression element-wise and returns a NumPy array.
```python
import numpy as np
import colss
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
colss.query("a + b + 7")
colss.query("a > 1 ? 100 : 0")       # ternary
colss.query("sqrt(a) + sin(b)")
colss.query("a ^ 2")                  # a squared
colss.query("a ^ 2 + b ^ 0.5")       # complex expression
```
---
### mean
Returns the arithmetic mean of the evaluated expression as a scalar.
```python
colss.mean("a")
colss.mean("a + b")
colss.mean("a ^ 2 + b")
```
---
### sum
Returns the sum of the evaluated expression as a scalar.
```python
colss.sum("a")
colss.sum("a * 2")
colss.sum("a + b")
```
---
### prod
Returns the product of all elements of the evaluated expression as a scalar.
```python
colss.prod("a")
colss.prod("a + 1")
colss.prod("a * b")
```
---
### median
Returns the median of the evaluated expression as a scalar. For even-length arrays, returns the average of the two middle values.
```python
colss.median("a")
colss.median("a + b")
colss.median("a ^ 2 + b")
```
---
### sd
Returns the population standard deviation of the evaluated expression as a scalar.
```python
colss.sd("a")
colss.sd("a + b")
colss.sd("sqrt(a) + b ^ 2")
```
---
### variance
Returns the population variance of the evaluated expression as a scalar.
```python
colss.variance("a")
colss.variance("a + b")
colss.variance("a ^ 2 - b")
```
---
## Using with Pandas
```python
import pandas as pd
import numpy as np
import colss
df = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [4.0, 5.0, 6.0]
})
a = df["a"].to_numpy(dtype=np.float64)
b = df["b"].to_numpy(dtype=np.float64)
df["c"] = colss.query("a + b + 7")
m       = colss.mean("a + b")
med     = colss.median("a")
s       = colss.sd("a + b")
```
---
## Operators
| Operator | Description |
|----------|-------------|
| `+` `-` `*` `/` | Arithmetic |
| `^` | Exponentiation — `a ^ 2` raises `a` to the power of 2 |
| `%` | Modulo |
| `>` `<` `>=` `<=` `==` `!=` | Comparison |
| `and` `or` `not` | Logical |
| `condition ? a : b` | Ternary |
---
## Available Functions
```
abs(x)       sqrt(x)      
log(x)       log10(x)     exp(x)
sin(x)       cos(x)       tan(x)
floor(x)     ceil(x)
min(x, y)    max(x, y)
```
---
## Notes
- `^` is exponentiation, not bitwise XOR.
- All arrays used in an expression must be the same length.
- You can use both scalar and vector at the same time.
- A NaN or Inf result raises an error identifying the first offending index.
