import numpy as np
import colss

# -----------------------------
# Scalar + Array
# -----------------------------

def test_query_scalar_array():
    a = np.array([1., 2., 3.])
    b = 5
    result = colss.query("a + b", a=a, b=b)
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

    result = colss.query(expr, a=a, b=b, c=c)
    expected = (a * 2.5 + b / 3.1 - np.sqrt(c)
                + np.sin(a) * np.cos(b)) / 2

    assert np.allclose(result, expected)


# -----------------------------
# Ternary operator
# -----------------------------

def test_ternary():
    a = np.array([0., 1., 2., 3.])
    result = colss.query("a > 1 ? 100 : 0", a=a)
    expected = np.where(a > 1, 100, 0).astype(float)
    assert np.allclose(result, expected)


# -----------------------------
# Logical expressions (safe numeric form)
# -----------------------------

def test_logical_expression():
    a = np.array([1., 2., 3., 4.])
    b = np.array([4., 3., 2., 1.])

    result = colss.query("((a > 2) and (b < 3)) ? 1 : 0", a=a, b=b)
    expected = np.where((a > 2) & (b < 3), 1.0, 0.0)

    assert np.allclose(result, expected)


# -----------------------------
# Nested math functions
# -----------------------------

def test_nested_functions():
    a = np.array([1., 2., 3., 4.])
    result = colss.query("exp(log(a)) + pow(a, 2)", a=a)
    expected = np.exp(np.log(a)) + np.power(a, 2)
    assert np.allclose(result, expected)


# -----------------------------
# Bitwise XOR (Caret) operator
# -----------------------------

def test_bitwise_xor():
    a = np.array([1., 2., 3., 4., 5.])
    result = colss.query("a^2", a=a)
    expected = (a.astype(np.int64) ^ 2).astype(float)
    assert np.allclose(result, expected)


# -----------------------------
# Removed constants check
# -----------------------------

def test_removed_constants():
    try:
        colss.query("pi")
        assert False, "Should have failed to compile unknown variable 'pi'"
    except Exception as e:
        assert "Unknown id/variable" in str(e)
    try:
        colss.query("e")
        assert False, "Should have failed to compile unknown variable 'e'"
    except Exception as e:
        assert "Unknown id/variable" in str(e)


# -----------------------------
# Multi-dimensional (N-dimensional) array support
# -----------------------------

def test_multidimensional_array():
    a = np.array([[1., 2.], [3., 4.]])
    b = np.array([[5., 6.], [7., 8.]])
    result = colss.query("a * b + 2", a=a, b=b)
    expected = a * b + 2
    assert result.shape == (2, 2)
    assert np.allclose(result, expected)
