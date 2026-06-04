# colss

Fast expression evaluator for NumPy.


**GitHub:** https://github.com/SivaPA08/colss

---

# Features

| Category      | Support               |
|---------------|-----------------------|
| Arithmetic    | `+ - * / %`           |
| Bitwise       | `& \| ~ << >> ^`      |
| Logical       | `&& \|\| !`           |
| Comparison    | `== != < <= > >=`     |
| Ternary       | `?:`                  |
| Trigonometric | `sin cos tan asin acos atan` |
| Logarithmic   | `log log10`           |
| Exponential   | `exp`                 |
| Root          | `sqrt`                |
| Rounding      | `floor ceil round`    |
| Utility       | `abs min max`         |

---

# Installation

```bash
pip install colss
```

---

# Usage

```python
import numpy as np
import colss as cs

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print(cs.query("a + b", a=a, b=b))
print(cs.query("sqrt(a) + sin(b)", a=a, b=b))
print(cs.query("a > b ? a : b", a=a, b=b))
print(cs.query("max(1,1,3,1,1)"))
```

---

# Syntax

```python
a + b
a - b
a * b
a / b
a % b
a ^ b

(a + b) * c

a > b
a <= b
a == b

(a > 0) && (b > 0)
(a > 0) || (b > 0)
!(a > 0)

a > b ? a : b
```

---

# Functions

```text
abs(x)
sqrt(x)
log(x)
log10(x)
exp(x)
sin(x)
cos(x)
tan(x)
asin(x)
acos(x)
atan(x)
floor(x)
ceil(x)
round(x)
min(...)
max(...)
```

`min()` and `max()` support multiple arguments.

```python
cs.query("max(1,1,3,1,1)")
cs.query("min(a,b,c,d)")
```

---

# Multidimensional Arrays

`colss` supports multidimensional arrays directly.

```python
import numpy as np
import colss as cs

a = np.random.rand(2, 3, 4)
b = np.random.rand(2, 3, 4)

res = cs.query("a + b", a=a, b=b)
print(res.shape)
```

---

# Notes

* Arrays in the same expression must have identical shapes.
* `query()` returns a NumPy array.
* For best performance, use `float64` and C-contiguous arrays.
* No constants are built in.
