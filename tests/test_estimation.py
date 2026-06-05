import numpy as np

from tvccointreg import estimation as est


def test_build_design_shape():
    T, K, q = 50, 3, 4
    X = np.random.randn(T, K)
    Z = np.random.randn(T, q)
    W = est.build_design(X, Z)
    assert W.shape == (T, K * q)
    # block j must equal X[:, j, None] * Z
    np.testing.assert_allclose(W[:, :q], X[:, [0]] * Z)


def test_variance_components_nonnegative():
    rng = np.random.default_rng(0)
    T, K = 200, 3
    X = rng.normal(size=(T, K))
    resid = rng.normal(size=T)
    s2 = est.variance_components(resid, X)
    assert s2.shape == (K,)
    assert np.all(s2 >= 0)


def test_irgls_recovers_constant_coefficients():
    # plain linear model with constant coefficients => TVC should be ~flat
    rng = np.random.default_rng(1)
    T = 400
    x = rng.normal(size=T)
    y = 2.0 + 1.5 * x + rng.normal(scale=0.1, size=T)
    X = np.column_stack([np.ones(T), x])
    Z = np.column_stack([np.ones(T), rng.normal(size=T)])  # const + 1 driver
    W = est.build_design(X, Z)
    fit = est.irgls(y, W, X)
    # intercept block: pi for const-coef = [2, 0]; slope block = [1.5, 0]
    assert fit.converged or fit.n_iter >= 1
    # reconstruct average coefficients
    q = 2
    b_const = (Z @ fit.pi[0:q]).mean()
    b_slope = (Z @ fit.pi[q:2 * q]).mean()
    assert abs(b_const - 2.0) < 0.1
    assert abs(b_slope - 1.5) < 0.1
