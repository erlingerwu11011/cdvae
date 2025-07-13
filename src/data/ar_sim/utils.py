import numpy as np
from numba import jit


@jit(nopython=True)
def eps_x_sample(n, cov_matrix, d_x):
    L = np.linalg.cholesky(cov_matrix)
    z = np.random.randn(n, d_x)
    return np.dot(z, L.T).reshape(n, d_x, 1)


@jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@jit(nopython=True)
def coef_y_re(w, p, coeff=1.2):
    if w == 0:
        return np.random.normal(0.1, 0.01, size=(1))
    elif w == 1:
        return np.random.normal(coeff, 0.01, size=(1))
    else:
        raise NotImplementedError()


@jit(nopython=True)
def coef_y_x(w, p):
    if w == 0:
        return np.random.normal(0.2, 0.01, size=(p))
    elif w == 1:
        return np.random.normal(0.0, 0.01, size=(p))
    else:
        raise NotImplementedError()


@jit(nopython=True)
def coef_y_w(w, p):
    if w == 0:
        return np.random.normal(0.1, 0.01, size=(p))
    elif w == 1:
        return np.random.normal(0.2, 0.01, size=(p))
    else:
        raise NotImplementedError()


@jit(nopython=True)
def coef_y(w, p):
    if w == 0:
        return np.random.normal(0.1, 0.01, size=(p))
    elif w == 1:
        return np.random.normal(0.2, 0.01, size=(p))
    else:
        raise NotImplementedError()


@jit(nopython=True)
def coef_w(treat_imb, t, p):
    return np.random.normal(0.2, 0.01, size=(p))


@jit(nopython=True)
def coef_w_x(treat_imb, t, d_vitals, p):
    return np.random.normal(treat_imb * np.sin(t / np.pi), 0.01, size=(d_vitals, p))


@jit(nopython=True)
def coef_w_y(treat_imb, t, p):
    return np.random.normal(0.02, 0.005, size=(p))


@jit(nopython=True)
def coef_x(d_vitals, p):
    return np.random.normal(2, 0.1, size=(d_vitals, p))


@jit(nopython=True)
def coef_x_w(p):
    return np.random.normal(0.5, 0.01, size=(p))


def initialize_data(n: int, d_x: int, p: int, cov_matrix: np.array):
    X_init = np.random.multivariate_normal(np.zeros(d_x), cov_matrix, n * p).reshape(n, d_x, -1)
    W_init = np.random.binomial(1, 0.5, size=(n, p))
    Y_init = np.random.normal(0, 1, size=(n, p))
    return X_init, W_init, Y_init
