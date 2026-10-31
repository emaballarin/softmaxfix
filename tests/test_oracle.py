"""Self-checks of the mpmath oracle: closed forms, derivative formulas against numerical differentiation."""

import numpy as np
import oracle
import pytest

CTX = oracle._CTX


def _dd_value(dd, idx):
    return CTX.mpf(float(dd[0][idx])) + CTX.mpf(float(dd[1][idx]))


def test_closed_forms():
    z = np.array([[0.0, 0.0], [3.0, -1.0]])
    ls = oracle.log_softmax(z)
    assert abs(_dd_value(ls, (0, 0)) + CTX.log(2)) < CTX.mpf(10) ** -30
    sig = 1 / (1 + CTX.exp(-4))
    assert abs(_dd_value(oracle.softmax(z), (1, 0)) - sig) < CTX.mpf(10) ** -30
    assert np.all(oracle.kl(z, z)[0] == 0.0)
    ent = -(sig * CTX.log(sig) + (1 - sig) * CTX.log(1 - sig))
    assert abs(_dd_value(oracle.soft_ce(z, z), 1) - ent) < CTX.mpf(10) ** -30


def test_masked_entries():
    ninf = -np.inf
    z = np.array([[1.0, ninf, 2.0]])
    ls = oracle.log_softmax(z)
    assert ls[0][0, 1] == -np.inf
    assert oracle.softmax(z)[0][0, 1] == 0.0
    # Teacher-masked class contributes nothing to soft CE or KL; values stay finite.
    zs = np.array([[0.0, 1.0, -1.0]])
    zt = np.array([[0.0, 1.0, ninf]])
    assert np.isfinite(oracle.soft_ce(zs, zt)[0]).all()
    assert np.isfinite(oracle.kl(zs, zt)[0]).all()
    assert oracle.grad_teacher_ce(zs, zt)[0][0, 2] == 0.0


def _num_grad(fn, row):
    """Central numerical gradient of a scalar mpmath function of a list of mpf."""
    base = [CTX.mpf(float(v)) for v in row]
    out = []
    for j in range(len(base)):

        def f(h, j=j):
            zz = list(base)
            zz[j] = zz[j] + h
            return fn(zz)

        out.append(CTX.diff(f, 0))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_derivative_formulas_match_numerical_differentiation(seed):
    rng = np.random.default_rng(np.random.SeedSequence([20261026, seed]))
    zs = rng.normal(size=(1, 4)) * 2
    zt = rng.normal(size=(1, 4)) * 2
    t = np.array([int(rng.integers(4))])
    p = rng.dirichlet(np.ones(4), size=1)
    v = rng.normal(size=(1, 4))
    tol = CTX.mpf(10) ** -25
    pm = [CTX.mpf(float(x)) for x in p[0]]
    vm = [CTX.mpf(float(x)) for x in v[0]]
    zt_m = [CTX.mpf(float(x)) for x in zt[0]]
    zs_m = [CTX.mpf(float(x)) for x in zs[0]]

    checks = {
        "grad_ce_int": (oracle.grad_ce_int(zs, t), lambda z: -oracle._log_softmax(z)[int(t[0])], zs[0]),
        "grad_ce_prob": (
            oracle.grad_ce_prob(zs, p),
            lambda z: -CTX.fsum(a * b for a, b in zip(pm, oracle._log_softmax(z))),
            zs[0],
        ),
        "log_softmax_vjp": (
            oracle.log_softmax_vjp(zs, v),
            lambda z: CTX.fsum(a * b for a, b in zip(vm, oracle._log_softmax(z))),
            zs[0],
        ),
        "softmax_vjp": (
            oracle.softmax_vjp(zs, v),
            lambda z: CTX.fsum(a * b for a, b in zip(vm, oracle._softmax(z))),
            zs[0],
        ),
        "grad_student": (
            oracle.grad_student(zs, zt),
            lambda z: -CTX.fsum(a * b for a, b in zip(oracle._softmax(zt_m), oracle._log_softmax(z))),
            zs[0],
        ),
        "grad_teacher_ce": (
            oracle.grad_teacher_ce(zs, zt),
            lambda z: -CTX.fsum(a * b for a, b in zip(oracle._softmax(z), oracle._log_softmax(zs_m))),
            zt[0],
        ),
        "grad_teacher_kl": (
            oracle.grad_teacher_kl(zs, zt),
            lambda z: CTX.fsum(
                a * (b - c) for a, b, c in zip(oracle._softmax(z), oracle._log_softmax(z), oracle._log_softmax(zs_m))
            ),
            zt[0],
        ),
    }
    for name, (dd, fn, row) in checks.items():
        num = _num_grad(fn, row)
        for j, nj in enumerate(num):
            assert abs(_dd_value(dd, (0, j)) - nj) < tol, name

    # Hessian of soft CE w.r.t. the student logits: numerical derivative of the analytic gradient q − p.
    hess = oracle.hessian(zs)
    for i in range(4):
        num = _num_grad(lambda z, i=i: oracle._softmax(z)[i], zs[0])
        for j, nj in enumerate(num):
            assert abs(_dd_value(hess, (0, 4 * i + j)) - nj) < tol


def test_ulp_and_errors():
    assert oracle.ulp(np.array([1.0]), "float32")[0] == 2.0**-23
    assert oracle.ulp(np.array([0.0]), "float32")[0] == 2.0**-149
    assert oracle.ulp(np.array([0.75]), "bfloat16")[0] == 2.0**-8
    assert oracle.ulp(np.array([1e-300]), "float32")[0] == 2.0**-149
    ref = (np.array([1.0, -np.inf, np.nan]), np.zeros(3))
    err = oracle.ulp_error(np.array([1.0 + 2.0**-23, -np.inf, np.nan]), ref, "float32")
    assert err.tolist() == [1.0, 0.0, 0.0]
    err = oracle.ulp_error(np.array([1.0, 0.0, 1.0]), ref, "float32")
    assert err[0] == 0.0 and np.isinf(err[1]) and np.isinf(err[2])
    nw = oracle.normwise_error(np.array([[1.0, 2.0**-20]]), (np.array([[1.0, 0.0]]), np.zeros((1, 2))), "float32")
    assert nw[0] == 8.0


def test_double_double_carries_extra_precision():
    z = np.array([[0.1, 0.2, 0.3]])
    hi, lo = oracle.log_softmax(z)
    exact = oracle._log_softmax([CTX.mpf(0.1), CTX.mpf(0.2), CTX.mpf(0.3)])
    for k in range(3):
        assert abs((CTX.mpf(hi[0, k]) + CTX.mpf(lo[0, k])) - exact[k]) < CTX.mpf(10) ** -30
        assert lo[0, k] != 0.0


def test_normwise_error_on_subnormal_rows():
    # A row whose largest reference is subnormal: scale·eps underflows to 0, which must not turn 0/0 into NaN.
    ref = (np.array([[4.2637e-310, -1.7145e-310]]), np.zeros((1, 2)))
    assert oracle.normwise_error(ref[0].copy(), ref, "float64")[0] == 0.0
    assert np.isinf(oracle.normwise_error(np.array([[1e-300, 0.0]]), ref, "float64")[0])
