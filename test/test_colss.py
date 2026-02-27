import numpy as np
import colss


# -----------------------------
# Basic reductions
# -----------------------------

def test_sigma():
    a = np.array([1., 2., 3., 4.])
    assert colss.sum("a") == 10.0


def test_mean():
    a = np.array([1., 2., 3., 4.])
    assert colss.mean("a") == 2.5


def test_prod():
    a = np.array([1., 2., 3., 4.])
    assert colss.prod("a") == 24.0


# -----------------------------
# Scalar + Array
# -----------------------------

def test_query_scalar_array():
    a = np.array([1., 2., 3.])
    b = 5
    result = colss.query("a + b")
    expected = a + b
    assert np.allclose(result, expected)


# -----------------------------
# Complex arithmetic expression
# -----------------------------

def test_complex_expression():
    a = np.array([1., 2., 3., 4.])
    b = np.array([4., 5., 6., 7.])
    c = np.array([2., 3., 4., 5.])

    expr = "(a * 2.5 + b / 3.1 - sqrt(c) + sin(a) * cos(b)) / 2"

    result = colss.query(expr)
    expected = (a * 2.5 + b / 3.1 - np.sqrt(c)
                + np.sin(a) * np.cos(b)) / 2

    assert np.allclose(result, expected)


# -----------------------------
# Ternary operator
# -----------------------------

def test_ternary():
    a = np.array([0., 1., 2., 3.])
    result = colss.query("a > 1 ? 100 : 0")
    expected = np.where(a > 1, 100, 0).astype(float)
    assert np.allclose(result, expected)


# -----------------------------
# Logical expressions (safe numeric form)
# -----------------------------

def test_logical_expression():
    a = np.array([1., 2., 3., 4.])
    b = np.array([4., 3., 2., 1.])

    result = colss.query("((a > 2) and (b < 3)) ? 1 : 0")
    expected = np.where((a > 2) & (b < 3), 1.0, 0.0)

    assert np.allclose(result, expected)


# -----------------------------
# Nested math functions
# -----------------------------

def test_nested_functions():
    a = np.array([1., 2., 3., 4.])
    result = colss.query("exp(log(a)) + pow(a, 2)")
    expected = np.exp(np.log(a)) + np.power(a, 2)
    assert np.allclose(result, expected)


# -----------------------------
# Multi-variable reduction
# -----------------------------

def test_sigma_complex():
    a = np.array([1., 2., 3., 4.])
    b = np.array([2., 2., 2., 2.])

    result = colss.sum("a * b + 3")
    expected = np.sum(a * b + 3)

    assert np.isclose(result, expected)


def test_var_basic():
    a = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    expected = np.var(a)
    result = colss.var("a")

    assert np.isclose(result, expected)


def test_sd_basic():
    a = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    expected = np.std(a)
    result = colss.sd("a")

    assert np.isclose(result, expected)

def test_median_odd_length():
    a = np.array([5., 1., 3., 2., 4.])
    expected = np.median(a)
    result = colss.median("a")
    assert np.isclose(result, expected)


def test_median_even_length():
    a = np.array([1., 2., 3., 4.])
    expected = np.median(a)
    result = colss.median("a")
    assert np.isclose(result, expected)


def test_median_negative_values():
    a = np.array([-5., -1., -3., -2., -4.])
    expected = np.median(a)
    result = colss.median("a")
    assert np.isclose(result, expected)


def test_median_repeated_values():
    a = np.array([2., 2., 2., 2., 2.])
    expected = np.median(a)
    result = colss.median("a")
    assert np.isclose(result, expected)


def test_median_expression_basic():
    a = np.array([1., 2., 3., 4., 5.])
    b = np.array([5., 4., 3., 2., 1.])

    expected = np.median(a + b)
    result = colss.median("a + b")

    assert np.isclose(result, expected)


def test_median_expression_complex():
    a = np.array([1., 2., 3., 4., 5.])

    expected = np.median(
        np.sin(a) +
        np.log(a + 1) +
        np.sqrt(a**2 + 3*a + 1) +
        np.cos(np.pi * a)
    )

    result = colss.median(
        "sin(a) + log(a+1) + sqrt(a^2 + 3*a + 1) + cos(pi*a)"
    )

    assert np.isclose(result, expected)


def test_median_random_large():
    a = np.random.rand(1000).astype(np.float64)

    expected = np.median(a)
    result = colss.median("a")

    assert np.isclose(result, expected)


def test_median_even_expression():
    a = np.array([1., 2., 3., 4.])

    expected = np.median(a * 2 + 1)
    result = colss.median("a * 2 + 1")

    assert np.isclose(result, expected)
