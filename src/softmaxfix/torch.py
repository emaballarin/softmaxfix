"""PyTorch back-end: drop-in replacements for the softmax family of ``torch.nn.functional`` and ``torch.nn``.

Replaces, with identical signatures and semantics (legacy ``dim=None`` rule, ``dtype=``, every ``cross_entropy``
option: class weights, ``ignore_index``, label smoothing, probability targets, K-dimensional inputs, legacy
``size_average`` / ``reduce``):

- ``F.log_softmax``, ``F.softmax``, ``F.softmin``, ``F.cross_entropy``
- ``nn.LogSoftmax``, ``nn.Softmax``, ``nn.Softmin``, ``nn.CrossEntropyLoss`` (subclasses: ``isinstance``,
  ``state_dict`` and ``repr`` are unchanged)

and adds two losses that take *teacher logits*, which is what removing Softmax Cancellation requires:

- ``soft_target_cross_entropy(input, target_logits)``: −Σ p log q per position, p = softmax(target), q = softmax(input)
- ``kl_div_with_logits(input, target_logits)``: KL(p ‖ q) per position

Same algorithms and derivative rules as ``softmaxfix.jax`` (see its docstring for the formulas and for the cases beyond
the paper's reference listings they cover; README.md, "Relation to the paper", maps the listings to these functions),
except the cotangent bound of ``rule="flush_proof"`` (below), which JAX cannot have, and the default rule:
``"accurate"`` here, since torch does not flush subnormals by default, ``"flush_proof"`` in JAX, since XLA CPU does.
float16 / bfloat16 inputs are computed in float32 on every device, so no
result is less accurate than native's on any device; outputs are in the input dtype. Each function keeps one tensor
for backward, as native does: softmax its output (its input when upcasting), log_softmax its input, from which it
recomputes the probabilities (from its output they would carry its rounding). Each runs several full-size passes
rather than one fused kernel. Backward and forward-mode rules are
written with differentiable operations, so double backward and ``torch.func`` transforms (``grad``, ``vmap``,
``jacrev``, ``jacfwd``, ``hessian``) work. TorchScript is not supported.

Numerical options: ``set_options(...)`` sets them process-wide, ``with options(...):`` for a block in the current
thread (as ``torch.no_grad`` and ``torch.autocast``), and ``get_options()`` reads those in force. They are read when a
function is called, and its backward and forward-mode derivatives use the options of that call, whatever they are
when ``backward()`` runs.

- ``exact_shift`` (default True): x − x_max is kept exactly, as hi + lo (TwoSum), and exponentiated as e^{hi} +
  e^{hi}·lo. Exponentiating the rounded shift turns its rounding into a relative error of about ε·|x_j − x_max|/2, which
  is the whole componentwise error of the values and of the cross-entropy gradient shared by native, by the paper's code
  and by this library with the option off. Measured worst cases, rounded → exact shift (CPU): fp64 ``wide`` softmax 306
  → 3.7 ulp, cross-entropy gradient 306 → 4.3 ulp, cross-entropy Hessian 74 → 3.6 ε; fp32 ``margin_bigk`` softmax 64 →
  6.7 ulp (CUDA 65 → 4.1). It costs extra full-size elementwise passes: in eager mode, forward + backward runs ×1.4–2.0
  the time with the rounded shift (fp32, RTX 3050 Ti and CPU; VALIDATION.md, "Cost"), which is most of what the defaults
  cost over this library's previous rules (×1.7–2.1) in log_softmax and cross-entropy; in softmax the centred backward
  adds about as much. Against native, log_softmax, softmax and cross-entropy run ×3.6–12 the time in fp32 and ×11–30 in
  bf16 on CUDA, where native keeps bf16 temporaries (VALIDATION.md). False restores the rounded shift (not in the
  soft-target support-mass correction, exact either way).
- ``rule`` (default ``"accurate"``): the derivative rules of softmax and log_softmax, and so of everything built on
  them (cross-entropy, the soft-target losses' second derivatives).

    - ``"accurate"``: per row, the exact rewriting that is accurate for it (``_ls_vjp``, ``_sm_vjp``, ``_ls_jvp``
      and ``_sm_jvp`` state the rules).
    - ``"flush_proof"``: ``"accurate"`` with the centred softmax construction formed at scale where its products
      could fall below the normal range (``_centred_z_scaled``; factors of exactly 1 on other rows, so they are
      computed as under ``"accurate"``). It exists for flush-to-zero arithmetic: XLA CPU (``softmaxfix.jax``), and
      torch CPU under ``torch.set_flush_denormal(True)``, where it removes the same loss (fp32 flush-band test rows:
      4 551 ε → 0). By default torch does not flush, and the option changes only the rows it scales, whose products
      it rounds as normal numbers instead of subnormals, tying with ``"accurate"`` on every measured key. The first
      derivatives of log_softmax and cross-entropy are unchanged; their second derivatives go through the softmax
      rule. A row is scaled only while its cotangent (or tangent), centred at the argmax, is below 2^T in magnitude
      on the classes of nonzero probability, so no scaled result overflows; on a row left unscaled for it, any product
      that could fall below the normal range is below 2^−40 of the row's largest (2^−459 in float64). ``softmaxfix.jax``
      has no such bound (its rule must stay linear in the tangent).
    - ``"paper"``: the paper's rules, those its code gets from autodiff, in every row: the argmax entry of a
      reverse-mode result is minus the sum of the others (zero-sum by construction), and forward mode is centred at the
      argmax. It cancels on flat rows (log_softmax VJP on ``flat``: 37–50 ε in fp32 and fp64, against 0.44–0.47 ε), and
      on JAX CPU an uncentred softmax VJP can lose up to 5.4·10⁴ ε to flushed subnormals (fp32 ``margin_bigk``,
      depending on how XLA compiles the sum; torch does not flush by default). It is the paper's rule, not its code: it
      runs on this library's kernels (the exact shift unless ``exact_shift=False``, the target-pivot cross-entropy
      gradient, the fixes listed in ``softmaxfix.jax``); a literal port of the paper's code is in
      ``tests/paper_torch.py`` and serves as a reference only.

  Integer-target cross-entropy pivots its gradient at the target under every rule: q_j off the target and
  −Σ_{j≠t} q_j on it, componentwise accurate and zero-sum by construction (Eq. 6's structure).

Measured over 20 seeds, 356 (regime, format, metric) keys per device, CPU and CUDA (VALIDATION.md, "Twenty-seed
comparisons"): the defaults are more accurate than this library's previous rules (rounded shift, softmax VJP pivoted
with r·v_max) on 87 / 88 keys and less accurate on 3 / 3 (and mixed on 1 / 2), single-row worst cases except a mean
shift of +0.10 ε (CPU, fp64 ``margin_bigk`` log_softmax Σg); against the paper's code, 120 keys better and 26 worse or
mixed, mostly the zero-sum residual Σg, which only the paper's rule makes exactly 0. Componentwise (5 / 3 seeds): the
gradient of q_max (softmax VJP at the one-hot cotangent of the argmax) stays within 7 ulp in every regime, where the
paper's code reaches 510 ulp and native 9·10¹⁵.

Reverse-mode results are **not projected onto Σ = 0**, although the exact VJPs sum to zero. Subtracting the row mean of
the computed gradient cuts the residual Σg (log_softmax worst 7.2 → 3.0 ε on fp64 ``safe``, CUDA), but it shifts every
entry by about ε·Σ|g|/K and so destroys the entries much smaller than that: the gradient of q_max reached 2·10¹⁵⁸ ulp on
fp64 ``wide`` (4.4 without), the log_softmax VJP 1.8·10⁵ ulp on fp32 ``margin_bigk`` (303 without). Rules and options
are the same as ``softmaxfix.jax``'s (but for that bound); JAX derives its reverse mode by transposing the forward-mode
rule, torch writes both, and XLA fuses the extra elementwise passes that torch eager runs as separate kernels: the exact
shift and the accurate rule against the previous rules cost ×0.95–1.02 under JAX jit on CPU and ×1.15–1.30 on GPU,
against torch eager's ×1.7–2.1.
"""

import contextlib
import math
import threading
from collections.abc import Generator
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import _reduction

__all__ = [
    "CrossEntropyLoss",
    "LogSoftmax",
    "Options",
    "Softmax",
    "Softmin",
    "cross_entropy",
    "get_options",
    "kl_div_with_logits",
    "log_softmax",
    "options",
    "set_options",
    "soft_target_cross_entropy",
    "softmax",
    "softmin",
]


# ----------------------------------------------------------------------------------------------- options
class Options(NamedTuple):
    """Numerical options (see the module docstring)."""

    exact_shift: bool = True
    rule: str = "accurate"


_RULES = ("accurate", "flush_proof", "paper")
_global_options = Options()
_scoped = threading.local()  # Each thread's innermost ``options`` block.


def _updated(current: Options, exact_shift: bool | None, rule: str | None) -> Options:
    if exact_shift is not None and not isinstance(exact_shift, bool):
        raise TypeError(f"exact_shift must be a bool, got {type(exact_shift).__name__}")
    if rule is not None and rule not in _RULES:
        raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
    return current._replace(**{k: v for k, v in (("exact_shift", exact_shift), ("rule", rule)) if v is not None})


def get_options() -> Options:
    """The numerical options in force: those of the innermost ``options`` block of this thread, else the
    process-wide ones."""
    return getattr(_scoped, "options", None) or _global_options


def set_options(*, exact_shift: bool | None = None, rule: str | None = None) -> Options:
    """Set the process-wide numerical options (``None`` leaves one unchanged) and return the previous process-wide
    ones. An enclosing ``options`` block still takes precedence in its thread until it exits."""
    global _global_options
    previous = _global_options
    _global_options = _updated(previous, exact_shift, rule)
    return previous


@contextlib.contextmanager
def options(*, exact_shift: bool | None = None, rule: str | None = None) -> Generator[Options]:
    """Set numerical options for the ``with`` block, in this thread (as ``torch.no_grad`` and ``torch.autocast``);
    the previous ones are restored on exit, exceptions included."""
    new = _updated(get_options(), exact_shift, rule)
    outer = getattr(_scoped, "options", None)
    _scoped.options = new
    try:
        yield new
    finally:
        _scoped.options = outer


def _compute_dtype(x: Tensor) -> torch.dtype:
    """float16 / bfloat16 are computed in float32 (on every device); other dtypes in their own precision."""
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


# --------------------------------------------------------------------------- forward kernels (no autograd)
# These run inside autograd.Function.forward, where autograd is off, and update their own temporaries in place.
def _exact_shift(x: Tensor, x_max: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """hi = fl(x − x_max), lo with hi + lo = x − x_max exactly (TwoSum, for any magnitudes; lo is 0 where hi is not
    finite), and e^{x − x_max} as e^{hi} + e^{hi}·lo. Four full-size temporaries and a transient."""
    hi = x - x_max
    bb = hi - x
    lo = hi - bb
    lo.neg_().add_(x)  # x − (hi − bb) as −(hi − bb) + x: the same rounding, in place (out= ops have no vmap rule).
    lo.sub_(bb.add_(x_max)).nan_to_num_(nan=0.0)  # (x − (hi − bb)) − (x_max + bb); NaN only where hi is ±inf or NaN.
    e = hi.exp()
    return hi, lo, e.add_(e * lo)


def _ls_fwd(x: Tensor, exact: bool) -> Tensor:
    """log_softmax over the last axis as (x − x_max) − log1p(Σ_{j≠max} e^{x_j − x_max}) (Eq. 5): the argmax's
    exponential is excluded from the sum, so nothing small is ever added to 1.0. With ``exact`` the shift is
    hi + lo and lo is added last. NaN for ±inf maxima, as native."""
    x_max, idx = x.max(-1, keepdim=True)
    if exact:
        out, lo, e = _exact_shift(x, x_max)
    else:
        out, lo = x - x_max, None
        e = out.exp()
    s = e.scatter_(-1, idx, 0.0).sum(-1, keepdim=True)
    del e
    out.sub_(torch.log1p(s) + (x_max - x_max))  # (x_max − x_max) is NaN for infinite maxima.
    return out if lo is None else out.add_(lo)


def _sm_fwd(x: Tensor, exact: bool) -> Tensor:
    """softmax over the last axis as e^{x − x_max} / Σ e^{x − x_max}, the shift exact with ``exact``."""
    x_max = x.amax(-1, keepdim=True)
    e = _exact_shift(x, x_max)[2] if exact else (x - x_max).exp_()
    return e.div_(e.sum(-1, keepdim=True) + (x_max - x_max))


# ------------------------------------------------------------------------------ derivative rules (autograd)
# Out of place throughout: an in-place update inside backward breaks double backward and torch.func.vmap.
class _ValueWithDerivativeOf(torch.autograd.Function):
    """``value``, with the derivative of ``proxy`` (equal to it up to rounding): a straight-through that, unlike
    ``detach()``, works under torch.func transforms and double backward."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value: Tensor, proxy: Tensor) -> Tensor:
        return value.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, g: Tensor):
        return None, g

    @staticmethod
    def jvp(ctx, d_value: Tensor | None, d_proxy: Tensor | None) -> Tensor:
        return d_proxy


def _rest(q: Tensor, idx: Tensor) -> Tensor:
    """1 − q_max as the sum of the other probabilities (no cancellation)."""
    return q.scatter(-1, idx, 0.0).sum(-1, keepdim=True)


def _scale_factors(q: Tensor, idx: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """ "flush_proof": per row, 2^−T and 2^T where the largest non-argmax probability is below 2^−(T+1) and the centred
    w (cotangent or tangent), |w_j − w_argmax| over the classes with q_j > 0, is below 2^T; T half the normal exponent
    range of q's format (63 in float32, 511 in float64); exactly 1 and 1 elsewhere. Exact powers of two: scaling by them
    is exact (no ``ldexp``). The bound keeps every scaled product below 2^T, so no scaled result overflows; classes of
    zero probability contribute exact zeros and are left out of it. On a row left unscaled for it, some class with
    q_k > 0 has |w_k − w_argmax| ≥ 2^T, so its product is at least 2^(T − 149) in float32 (2^(T − 1074) in float64),
    and any product that could fall below the normal range is below 2^−40 of it (2^−459)."""
    t = -int(math.log2(torch.finfo(q.dtype).tiny)) // 2
    centred = torch.where(q > 0, (w - w.gather(-1, idx)).abs(), 0.0)
    scaled = (q.scatter(-1, idx, 0.0).amax(-1, keepdim=True) < 2.0 ** (-t - 1)) & (
        centred.amax(-1, keepdim=True) < 2.0**t
    )
    one = torch.ones_like(q[..., :1])
    return torch.where(scaled, 2.0**-t, one), torch.where(scaled, 2.0**t, one)


def _centred_z_scaled(q: Tensor, idx: Tensor, w: Tensor) -> Tensor:
    """The centred-Z construction of the softmax rule on w (cotangent or tangent), with the products formed at 2^T
    their size where they could fall below the normal range: input × 2^−T, q × 2^{2T}, output × 2^−T, so that a
    flush-to-zero arithmetic cannot drop the products the argmax entry sums. Rows not scaled get factors of exactly
    1, so they are the centred-Z construction itself."""
    dn, up = _scale_factors(q, idx, w)
    wa = w * dn
    xa = (wa - wa.gather(-1, idx)).scatter(-1, idx, 0.0)
    ys = (q * up * up) * (xa - (q * xa).sum(-1, keepdim=True))
    return dn * ys.scatter(-1, idx, -ys.scatter(-1, idx, 0.0).sum(-1, keepdim=True))


def _ls_vjp(q: Tensor, idx: Tensor, v: Tensor, rule: str) -> Tensor:
    """vᵀ J_logsoftmax = v − q·Σv; the argmax entry depends on ``rule``.

    "accurate": in rows with q_max > ½ the argmax entry is r·Σv − Σ_{j≠max} v_j with r = Σ_{j≠max} q_j = 1 − q_max
    (form C): v_max − q_max·Σv loses r once q_max rounds to 1 (Softmax Collapse), and (v_max − Σv) + Σv·r carries the
    rounding of Σv at the size of v_max. Not projected onto Σ = 0 (module docstring).
    "paper": the argmax entry is −Σ_{j≠max} g_j in every row (form Z, what autodiff of Eq. 5 computes): zero-sum by
    construction, but it cancels on flat rows whose target is not the argmax.
    One-hot cotangents (cross-entropy through nll_loss) take the target-pivot path under every rule."""
    s = v.sum(-1, keepdim=True)
    g = v - q * s
    if rule == "paper":
        g = g.scatter(-1, idx, -g.scatter(-1, idx, 0.0).sum(-1, keepdim=True))
    else:
        centred = s * _rest(q, idx) - v.scatter(-1, idx, 0.0).sum(-1, keepdim=True)
        g = g.scatter(-1, idx, torch.where(q.gather(-1, idx) > 0.5, centred, g.gather(-1, idx)))
    # One-hot rows (cross-entropy through nll_loss), nonzero at p: −q_j·v_p off p and v_p·Σ_{j≠p} q_j on it, a sum of
    # positive terms, so accurate at any margin and on flat rows, and zero-sum by construction (Eq. 6's structure).
    # The pivot's own derivative is that of v_p·(1 − q_p), −v_p·dq_p (the softmax rule), not v_p·Σ_{j≠p} dq_j, which
    # would cancel Hessian rows into each other (measured: never better, sometimes worse).
    nz = v != 0
    p_idx = nz.int().argmax(-1, keepdim=True)
    v_p, q_p = v.gather(-1, p_idx), q.gather(-1, p_idx)
    rest_p = _ValueWithDerivativeOf.apply(_rest(q, p_idx), -q_p)
    g_onehot = (-v_p * q).scatter(-1, p_idx, v_p * rest_p)
    return torch.where(nz.sum(-1, keepdim=True) == 1, g_onehot, g)


def _sm_vjp(q: Tensor, idx: Tensor, v: Tensor, rule: str) -> Tensor:
    """vᵀ J_softmax = q ⊙ (v − ⟨q, v⟩); the argmax entry and the cotangent depend on ``rule``.

    "flush_proof": "accurate" with the centred-Z construction formed at scale (``_centred_z_scaled``); identical to
    "accurate" on rows whose non-argmax probabilities are not below 2^−(T+1).

    "accurate": in rows with q_max > ½ the cotangent is first centred, v → v − v_max (exact: J_softmax maps constants
    to 0), which removes the cancellation between large, nearly equal cotangent entries (fp32 ``wide``: worst 410 ε
    with r·v_max at the argmax, 151 ε without centring, 1.8 ε centred; numpy probe); the argmax entry is then −Σ of
    the others, accurate once the cotangent is centred. Rows with q_max ≤ ½ keep q ⊙ (v − ⟨q, v⟩): centring harms flat
    rows (Σg 467 ε). Not projected onto Σ = 0 (module docstring).
    "paper": u − q·Σu with u = q ⊙ v (the log_softmax rule on u, as autodiff of exp ∘ log_softmax computes it, and as
    JAX's transposed paper rule does), the argmax entry −Σ of the others, in every row, uncentred. On flat rows this
    order is markedly more accurate than q ⊙ (v − ⟨q, v⟩) under the Z pivot (fp32 ``flat``, mean 7–15 ε over 4 seeds
    against 36.5 ε at the suite's seed; the paper's code 17–25 ε)."""
    if rule == "paper":
        u = q * v
        g = u - q * u.sum(-1, keepdim=True)
        return g.scatter(-1, idx, -g.scatter(-1, idx, 0.0).sum(-1, keepdim=True))
    g = q * (v - (q * v).sum(-1, keepdim=True))
    if rule == "flush_proof":
        return torch.where(q.gather(-1, idx) > 0.5, _centred_z_scaled(q, idx, v), g)
    vc = (v - v.gather(-1, idx)).scatter(-1, idx, 0.0)  # Exactly 0 at the argmax.
    gc = q * (vc - (q * vc).sum(-1, keepdim=True))
    gc = gc.scatter(-1, idx, -gc.scatter(-1, idx, 0.0).sum(-1, keepdim=True))
    return torch.where(q.gather(-1, idx) > 0.5, gc, g)


def _ls_jvp(q: Tensor, idx: Tensor, dx: Tensor, rule: str) -> Tensor:
    """J_logsoftmax ẋ: centred at the argmax, (ẋ − ẋ_max) − Σ_{j≠max} q_j (ẋ_j − ẋ_max), in rows with q_max > ½
    ("accurate") or in every row ("paper"); ẋ − ⟨q, ẋ⟩ elsewhere. The argmax slot of the centred difference is set
    to 0, never computed as ẋ_max − ẋ_max."""
    w = (dx - dx.gather(-1, idx)).scatter(-1, idx, 0.0)
    centred = w - (q.scatter(-1, idx, 0.0) * w).sum(-1, keepdim=True)
    if rule == "paper":
        return centred
    plain = dx - (q * dx).sum(-1, keepdim=True)
    return torch.where(q.gather(-1, idx) > 0.5, centred, plain)


def _sm_jvp(q: Tensor, idx: Tensor, dx: Tensor, rule: str) -> Tensor:
    """J_softmax ẋ = q ⊙ (ẋ − ⟨q, ẋ⟩). "accurate": in rows with q_max > ½ the tangent is centred, ẋ → ẋ − ẋ_max, and
    the argmax entry is −Σ of the others (the same construction as the reverse rule, as in JAX's self-transposed
    rule); elsewhere
    q ⊙ (ẋ − ⟨q, ẋ⟩). "flush_proof": the same, the centred construction formed at scale (``_centred_z_scaled``).
    "paper": q ⊙ the centred log_softmax tangent, in every row."""
    if rule == "paper":
        return q * _ls_jvp(q, idx, dx, rule)
    if rule == "flush_proof":
        plain = q * (dx - (q * dx).sum(-1, keepdim=True))
        return torch.where(q.gather(-1, idx) > 0.5, _centred_z_scaled(q, idx, dx), plain)
    xc = (dx - dx.gather(-1, idx)).scatter(-1, idx, 0.0)
    y = q * (xc - (q * xc).sum(-1, keepdim=True))
    y = y.scatter(-1, idx, -y.scatter(-1, idx, 0.0).sum(-1, keepdim=True))
    return torch.where(q.gather(-1, idx) > 0.5, y, q * (dx - (q * dx).sum(-1, keepdim=True)))


class _LogSoftmax(torch.autograd.Function):
    """log_softmax over the last axis, returned in the compute dtype, with a collapse-free derivative."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x: Tensor, exact: bool, rule: str) -> Tensor:
        return _ls_fwd(x.to(_compute_dtype(x)), exact)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, ctx.exact, ctx.rule = inputs
        ctx.in_dtype = x.dtype
        # One tensor, the input: q = e^{out} would carry the rounding of out at magnitude |log q| (≈ ε·|log q|
        # relative), which the paper's autodiff avoids; recomputing q from x keeps the derivative as accurate.
        ctx.save_for_backward(x)
        ctx.save_for_forward(x)

    @staticmethod
    def _parts(ctx) -> tuple[Tensor, Tensor]:
        (x,) = ctx.saved_tensors
        q = _Softmax.apply(x, ctx.exact, ctx.rule)
        return q, q.argmax(-1, keepdim=True)

    @staticmethod
    def backward(ctx, v: Tensor):
        q, idx = _LogSoftmax._parts(ctx)
        return _ls_vjp(q, idx, v.to(q.dtype), ctx.rule).to(ctx.in_dtype), None, None

    @staticmethod
    def jvp(ctx, dx: Tensor, _exact, _rule) -> Tensor:
        q, idx = _LogSoftmax._parts(ctx)
        return _ls_jvp(q, idx, dx.to(q.dtype), ctx.rule)


class _Softmax(torch.autograd.Function):
    """softmax over the last axis, returned in the compute dtype, with a collapse-free derivative."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x: Tensor, exact: bool, rule: str) -> Tensor:
        return _sm_fwd(x.to(_compute_dtype(x)), exact)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, ctx.exact, ctx.rule = inputs
        ctx.upcast, ctx.in_dtype = x.dtype != output.dtype, x.dtype
        saved = x if ctx.upcast else output
        ctx.save_for_backward(saved)
        ctx.save_for_forward(saved)

    @staticmethod
    def _parts(ctx) -> tuple[Tensor, Tensor]:
        (saved,) = ctx.saved_tensors
        q = _Softmax.apply(saved, ctx.exact, ctx.rule) if ctx.upcast else saved
        return q, q.argmax(-1, keepdim=True)

    @staticmethod
    def backward(ctx, v: Tensor):
        q, idx = _Softmax._parts(ctx)
        return _sm_vjp(q, idx, v.to(q.dtype), ctx.rule).to(ctx.in_dtype), None, None

    @staticmethod
    def jvp(ctx, dx: Tensor, _exact, _rule) -> Tensor:
        q, idx = _Softmax._parts(ctx)
        return _sm_jvp(q, idx, dx.to(q.dtype), ctx.rule)


# ------------------------------------------------------------------------------------------ soft targets
def _expm1mx(x: Tensor) -> Tensor:
    """e^x − 1 − x without cancellation: Taylor series for |x| < 0.5, enough terms for the dtype's
    precision (x¹⁶/16! in float64, x⁹/9! otherwise); the direct difference elsewhere. Forward use only."""
    small = x.abs() < 0.5
    xs = torch.where(small, x, 0.0)
    acc = torch.ones_like(xs)
    for n in range(16 if x.dtype == torch.float64 else 9, 2, -1):  # x²/2 · (1 + x/3 · (1 + x/4 · (…))).
        acc.mul_(xs).div_(n).add_(1)
    direct = torch.expm1(x).sub_(x)
    return torch.where(small, acc.mul_(xs).mul_(xs).div_(2), direct)


def _log1pmx(y: Tensor) -> Tensor:
    """log(1 + y) − y without cancellation: Taylor series to y¹⁸/18 for |y| < 0.1, direct difference elsewhere."""
    small = y.abs() < 0.1
    ys = torch.where(small, y, 0.0)
    acc = torch.zeros_like(ys)
    for n in range(18, 1, -1):  # y² · (−1/2 + y·(1/3 + y·(−1/4 + …))).
        acc = (-1) ** (n + 1) / n + ys * acc
    yl = torch.where(small, 1.0, y)
    return torch.where(small, ys * ys * acc, torch.log1p(yl) - yl)


def _log1m(x: Tensor, log_comp: Tensor) -> Tensor:
    """log(1 − x): log1p(−x) below ½, else the caller's log(1 − x), evaluated from the complement's own terms."""
    low = x < 0.5
    return torch.where(low, torch.log1p(-torch.where(low, x, 0.0)), log_comp)


def _two_diff(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """hi = fl(a − b) and lo with hi + lo = a − b exactly (TwoSum), out of place (differentiable); lo is 0 where hi is
    not finite."""
    hi = a - b
    bb = hi - a
    lo = (a - (hi - bb)) - (b + bb)
    return hi, torch.where(torch.isfinite(hi), lo, 0.0)


def _lse_parts(z: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """log Σ_{j∈mask} e^{z_j} over the last axis as m + log1p(s) (keepdim): m the masked maximum, s the sum of the
    other masked e^{z_j − m} with the shift exact (Eq. 5's form), and whether the mask is empty there (m and log1p(s)
    are then 0, for the caller to replace). A mask whose logits are all −inf gives m = −inf and log1p(s) = 0, computed
    on zeros as the empty one; a +inf maximum gives log1p(s) = 0, the sum being +inf whatever s is. Out of place: it
    may run under autograd."""
    empty = ~mask.any(-1, keepdim=True)
    dead = ~(mask & (z != -math.inf)).any(-1, keepdim=True)  # The empty mask included.
    za = torch.where(dead, 0.0, torch.where(mask, z, -math.inf))
    m, idx = za.max(-1, keepdim=True)
    hi, lo = _two_diff(za, m)
    e = hi.exp()
    e = e + e * lo
    l1p = torch.log1p(torch.where(mask, e, 0.0).scatter(-1, idx, 0.0).sum(-1, keepdim=True))
    return torch.where(dead & ~empty, -math.inf, m), torch.where(dead | (m == math.inf), 0.0, l1p), empty


def _log_mass(z: Tensor, mask: Tensor) -> Tensor:
    """log Σ_{j∈mask} softmax(z)_j from the logits: exact where the probabilities themselves underflow. Both sums in Eq.
    5's form with the exact shift, the difference of their maxima exact (TwoSum): worst 97 → 2.0–3.4 ulp against the
    difference of two logsumexps on wide rows (VALIDATION.md, "Probes cited in the code"). An empty mask, or one on −inf
    logits only, gives −inf, a row without a finite logit NaN, as that difference did."""
    m_a, l_a, empty_a = _lse_parts(z, mask)
    m_z, l_z, empty_z = _lse_parts(z, z != -math.inf)
    hi, lo = _two_diff(m_a, m_z)
    out = (hi + (l_a - l_z)) + lo
    return torch.where(empty_z, math.nan, torch.where(empty_a, -math.inf, out))


def _compensated_b(z_s: Tensor, z_t: Tensor, both: Tensor, k: Tensor) -> Tensor:
    """b = δ − δ_k* with δ = z_S − z_T on classes both distributions support, exact 0 at the teacher's mode k*.

    δ is formed with its exact rounding remainder (TwoSum), so a common offset between student and teacher logits
    cancels exactly instead of after rounding at the offset's magnitude.
    """
    zs0, zt0 = torch.where(both, z_s, 0.0), torch.where(both, z_t, 0.0)
    delta = zs0 - zt0
    bv = delta - zs0
    err = (zs0 - (delta - bv)) - (zt0 + bv)  # delta + err == zs0 − zt0 exactly.
    return ((delta - delta.gather(-1, k)) + (err - err.gather(-1, k))).scatter(-1, k, 0.0)


def _support(z_s: Tensor, z_t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(supported by both, by the student, rows where every teacher class has student mass, i.e. finite loss), read
    from the logits: a class whose probability underflows still carries mass, and its KL term is finite."""
    pos_p, pos_q = z_t > -math.inf, z_s > -math.inf
    return pos_p & pos_q, pos_q, ~(pos_p & ~pos_q).any(-1, keepdim=True)


def _close_split(z_s: Tensor, q: Tensor, p: Tensor, b: Tensor, both: Tensor, pos_q: Tensor, supported: Tensor):
    """Classes within |b| ≤ 1 of the teacher's mode (log space) and the rest (own probabilities), with the sums,
    the renormalised teacher weights on the close set, d (b centred under them) and log(1 − Q_far), log(1 − P_far);
    log(1 − Q_far) = log Q_close is taken in log space when the close set holds the smaller part of q."""
    close = both & (b.abs() <= 1)
    far = pos_q & ~close
    p_c = torch.where(close, p, 0.0)
    p_close = p_c.sum(-1, keepdim=True)
    p_far = torch.where(far, p, 0.0).sum(-1, keepdim=True)
    q_far = torch.where(far, q, 0.0).sum(-1, keepdim=True)
    pt = p_c / torch.where(supported, p_close, 1.0)
    d = torch.where(close, b - (pt * b).sum(-1, keepdim=True), 0.0)
    lq = _log1m(q_far, _log_mass(z_s, close))
    lp = _log1m(p_far, torch.log(torch.where(supported, p_close, 1.0)))  # P_close ≥ p_k* ≥ 1/K: never underflows.
    return close, far, pt, d, p_close, p_far, lq, lp


def _student_grad_fwd(z_s: Tensor, z_t: Tensor, exact: bool) -> Tensor:
    """q − p (Eq. 9 on the close set, with the HTML revision's |s| < 1 fallback and the zero-sum correction)."""
    q, p = _sm_fwd(z_s, exact), _sm_fwd(z_t, exact)
    both, pos_q, supported = _support(z_s, z_t)
    b = _compensated_b(z_s, z_t, both, z_t.argmax(-1, keepdim=True))
    close, _, pt, d, _, _, lq, lp = _close_split(z_s, q, p, b, both, pos_q, supported)
    del b
    s = d.sub_(torch.log1p((pt * torch.expm1(d)).sum(-1, keepdim=True)) - lq + lp)  # log(q/p) on the close set.
    del pt
    small = supported & close & (s.abs() < 1)
    u = torch.where(small, p * torch.expm1(s.masked_fill_(~small, 0.0)), q - p)
    return u.sub_(q * u.sum(-1, keepdim=True))


def _kl_fwd(z_s: Tensor, z_t: Tensor, exact: bool) -> Tensor:
    """KL(p ‖ q) = P_close·(log Σ p̃·e^d − Σ p̃·d − log(1 − Q_far) + log(1 − P_far)) + Σ_far p·log(p/q), with the
    far terms log(p_k/q_k) = log(p_k*/q_k*) − b_k; +inf on rows whose loss is infinite."""
    q, p = _sm_fwd(z_s, exact), _sm_fwd(z_t, exact)
    both, pos_q, supported = _support(z_s, z_t)
    k = z_t.argmax(-1, keepdim=True)
    b = _compensated_b(z_s, z_t, both, k)
    _, far, pt, d, p_close, _, lq, lp = _close_split(z_s, q, p, b, both, pos_q, supported)
    del q, both, pos_q
    resid = (pt * d).sum(-1, keepdim=True)
    d_star = d.gather(-1, k)
    s_prime = (pt * _expm1mx(d)).sum(-1, keepdim=True)
    del pt, d
    y = s_prime + resid
    near = y < 1
    kl_core = torch.where(near, s_prime + _log1pmx(torch.where(near, y, 0.0)), torch.log1p(y) - resid)
    r_star = torch.log1p(y) - d_star - lq + lp  # log(p/q) at the teacher's mode.
    # Σ_far p_k·log(p_k/q_k) term by term: r*·P_far − Σ_far p_k·b_k would cancel two O(|b|) sums into the KL.
    far_terms = torch.where(far, p * (r_star - b), 0.0).sum(-1, keepdim=True)
    kl = p_close * (kl_core - lq + lp) + far_terms
    return torch.where(supported, kl, math.inf)[..., 0]


def _ce_fwd(z_s: Tensor, z_t: Tensor, exact: bool) -> Tensor:
    """−Σ p log q, with 0·log 0 := 0."""
    p = _sm_fwd(z_t, exact)
    return -p.mul_(_ls_fwd(z_s, exact).masked_fill_(p == 0, 0.0)).sum(-1)


def _teacher_grad(z_s: Tensor, z_t: Tensor, kl: bool, exact: bool, rule: str) -> Tensor:
    """d/dz_T from logit identities: −p ⊙ (z_S − ⟨p, z_S⟩) for CE, −p ⊙ (b − ⟨p, b⟩) for KL (NaN where KL = ∞).
    Differentiable (used inside backward, which double backward differentiates)."""
    p = _Softmax.apply(z_t, exact, rule)
    k = z_t.argmax(-1, keepdim=True)
    if not kl:
        zs_k = z_s.gather(-1, k)
        a = torch.where(p > 0, z_s - torch.where(zs_k.isfinite(), zs_k, 0.0), 0.0)
        return -p * (a - (p * a).sum(-1, keepdim=True))
    with torch.no_grad():
        both, _, supported = _support(z_s, z_t)
    b = _compensated_b(z_s, z_t, both, k)
    return torch.where(supported, -p * (b - (p * b).sum(-1, keepdim=True)), math.nan)


class _StudentGrad(torch.autograd.Function):
    """q − p from the log-space evaluation. Its own derivative is that of q − p through the softmax rule:
    differentiating the log-space formula instead cancels O(1) terms into e^{−m}-sized curvature."""

    generate_vmap_rule = True

    @staticmethod
    def forward(z_s: Tensor, z_t: Tensor, exact: bool, rule: str) -> Tensor:
        return _student_grad_fwd(z_s, z_t, exact)

    @staticmethod
    def setup_context(ctx, inputs, output):
        z_s, z_t, ctx.exact, ctx.rule = inputs
        ctx.save_for_backward(z_s, z_t)
        ctx.save_for_forward(z_s, z_t)

    @staticmethod
    def backward(ctx, v: Tensor):
        z_s, z_t = ctx.saved_tensors
        q, p = _Softmax.apply(z_s, ctx.exact, ctx.rule), _Softmax.apply(z_t, ctx.exact, ctx.rule)
        i_q, i_p = q.argmax(-1, keepdim=True), p.argmax(-1, keepdim=True)
        return _sm_vjp(q, i_q, v, ctx.rule), -_sm_vjp(p, i_p, v, ctx.rule), None, None

    @staticmethod
    def jvp(ctx, dz_s: Tensor | None, dz_t: Tensor | None, _exact, _rule) -> Tensor:
        z_s, z_t = ctx.saved_tensors
        out = torch.zeros_like(z_s)
        if dz_s is not None:
            q = _Softmax.apply(z_s, ctx.exact, ctx.rule)
            out = out + _sm_jvp(q, q.argmax(-1, keepdim=True), dz_s, ctx.rule)
        if dz_t is not None:
            p = _Softmax.apply(z_t, ctx.exact, ctx.rule)
            out = out - _sm_jvp(p, p.argmax(-1, keepdim=True), dz_t, ctx.rule)
        return out


class _SoftTarget(torch.autograd.Function):
    """Per-row soft-target CE (``kl=False``) or KL (``kl=True``); forward computes the value only."""

    generate_vmap_rule = True

    @staticmethod
    def forward(z_s: Tensor, z_t: Tensor, kl: bool, exact: bool, rule: str) -> Tensor:
        cdt = _compute_dtype(z_s)
        zs, zt = z_s.to(cdt), z_t.to(cdt)
        return (_kl_fwd(zs, zt, exact) if kl else _ce_fwd(zs, zt, exact)).to(z_s.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output):
        z_s, z_t, ctx.kl, ctx.exact, ctx.rule = inputs
        ctx.save_for_backward(z_s, z_t)
        ctx.save_for_forward(z_s, z_t)

    @staticmethod
    def _grads(ctx) -> tuple[Tensor, Tensor]:
        z_s, z_t = ctx.saved_tensors
        cdt = _compute_dtype(z_s)
        zs, zt = z_s.to(cdt), z_t.to(cdt)
        return (
            _StudentGrad.apply(zs, zt, ctx.exact, ctx.rule),
            _teacher_grad(zs, zt, ctx.kl, ctx.exact, ctx.rule),
        )

    @staticmethod
    def backward(ctx, g: Tensor):
        z_s, z_t = ctx.saved_tensors
        g_s, g_t = _SoftTarget._grads(ctx)
        gc = g.to(g_s.dtype).unsqueeze(-1)
        return (gc * g_s).to(z_s.dtype), (gc * g_t).to(z_t.dtype), None, None, None

    @staticmethod
    def jvp(ctx, dz_s: Tensor | None, dz_t: Tensor | None, _kl, _exact, _rule):
        z_s, _ = ctx.saved_tensors
        g_s, g_t = _SoftTarget._grads(ctx)
        out = torch.zeros(g_s.shape[:-1], dtype=g_s.dtype, device=g_s.device)
        if dz_s is not None:
            out = out + (g_s * dz_s.to(g_s.dtype)).sum(-1)
        if dz_t is not None:
            out = out + (g_t * dz_t.to(g_t.dtype)).sum(-1)
        return out.to(z_s.dtype)


# ------------------------------------------------------------------------------------------------ public API
def _over_dim(fn, x: Tensor, dim: int) -> Tensor:
    """Apply a last-axis Function along ``dim`` with the current options: contiguous output in the input dtype, as
    native. A 0-d input is a single class; an empty class axis is returned as is."""
    if x.dim() == 0:
        if dim not in (0, -1):
            raise IndexError(f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})")
        return _over_dim(fn, x.unsqueeze(0), 0).squeeze(0)
    if x.shape[dim] == 0:
        return x.clone()
    opts = get_options()
    return fn(x.movedim(dim, -1), opts.exact_shift, opts.rule).to(x.dtype).movedim(-1, dim).contiguous()


def _check_floating(x: Tensor, name: str) -> None:
    if not x.is_floating_point():
        raise RuntimeError(f"{name}: expected a floating-point tensor, got {x.dtype}")


def log_softmax(
    input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: torch.dtype | None = None
) -> Tensor:
    """Drop-in for :func:`torch.nn.functional.log_softmax`, accurate at any logit margin."""
    if dim is None:
        dim = F._get_softmax_dim("log_softmax", input.dim(), _stacklevel)
    if dtype is not None:
        input = input.to(dtype)
    _check_floating(input, "log_softmax")
    return _over_dim(_LogSoftmax.apply, input, dim)


def softmax(input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: torch.dtype | None = None) -> Tensor:
    """Drop-in for :func:`torch.nn.functional.softmax`, with a collapse-free gradient (and, with the exact shift,
    a value accurate to a few ulp where native's carries the shift's rounding)."""
    if dim is None:
        dim = F._get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is not None:
        input = input.to(dtype)
    _check_floating(input, "softmax")
    return _over_dim(_Softmax.apply, input, dim)


def softmin(input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: torch.dtype | None = None) -> Tensor:
    """Drop-in for :func:`torch.nn.functional.softmin`, i.e. :func:`softmax` of ``−input``."""
    if dim is None:
        dim = F._get_softmax_dim("softmin", input.dim(), _stacklevel)
    if dtype is not None:
        input = input.to(dtype)
    _check_floating(input, "softmin")
    return _over_dim(_Softmax.apply, -input, dim)


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """Drop-in for :func:`torch.nn.functional.cross_entropy`, accurate at any logit margin.

    Built as the fixed :func:`log_softmax` followed by :func:`torch.nn.functional.nll_loss`, with label smoothing and
    probability targets reproduced exactly as ATen composes them, so every option keeps native semantics. For soft
    probability targets the gradient (Σp)·q − p still subtracts two rounded probabilities; pass teacher logits to
    :func:`soft_target_cross_entropy` to avoid that.
    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    if not 0.0 <= label_smoothing <= 1.0:
        raise RuntimeError(f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}")
    class_dim = 0 if input.dim() == 1 else 1
    n_classes = input.shape[class_dim]
    lsm = log_softmax(input, class_dim)
    w_shape = [1] * input.dim()
    w_shape[class_dim] = n_classes
    if input.shape == target.shape:  # ATen treats equal shapes as class probabilities.
        if not target.is_floating_point():
            raise RuntimeError(f"Expected floating point type for target with class probabilities, got {target.dtype}")
        if ignore_index >= 0:
            raise RuntimeError("ignore_index is not supported for floating point target")
        if label_smoothing > 0:
            target = target * (1 - label_smoothing) + label_smoothing / n_classes
        prod = lsm * target if weight is None else lsm * target * weight.view(w_shape)
        if reduction == "none":
            return -prod.sum(class_dim)
        if reduction == "sum":
            return -prod.sum()
        if reduction == "mean":
            return -prod.sum() / (input.numel() / n_classes) if input.numel() else -prod.sum() * math.nan
        raise ValueError(f"{reduction} is not a valid value for reduction")
    nll = F.nll_loss(lsm, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    if label_smoothing == 0:
        return nll
    ignored = target == ignore_index
    smooth = -(lsm if weight is None else lsm * weight.view(w_shape)).sum(class_dim).masked_fill(ignored, 0.0)
    if reduction == "none":
        ret = smooth
    elif reduction == "sum":
        ret = smooth.sum()
    elif weight is None:
        ret = smooth.sum() / (~ignored).sum()
    else:
        ret = smooth.sum() / weight[target.masked_select(~ignored)].sum()
    return (1 - label_smoothing) * nll + ret * (label_smoothing / n_classes)


def _soft_target(input: Tensor, target_logits: Tensor, kl: bool, reduction: str) -> Tensor:
    _check_floating(input, "input")
    _check_floating(target_logits, "target_logits")
    if input.shape != target_logits.shape:
        raise ValueError(
            f"input and target_logits must have the same shape, got {input.shape} and {target_logits.shape}"
        )
    dtype = torch.promote_types(input.dtype, target_logits.dtype)
    z_s, z_t = input.to(dtype), target_logits.to(dtype)
    class_dim = 0 if input.dim() == 1 else 1
    z_s, z_t = z_s.movedim(class_dim, -1), z_t.movedim(class_dim, -1)
    opts = get_options()
    loss = (
        _SoftTarget.apply(z_s, z_t, kl, opts.exact_shift, opts.rule)
        if z_s.shape[-1] > 0
        else torch.zeros(z_s.shape[:-1], dtype=dtype, device=z_s.device)
    )
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    if reduction == "batchmean" and kl:
        return loss.sum() / (input.shape[0] if input.dim() > 1 else 1)
    raise ValueError(f"{reduction} is not a valid value for reduction")


def soft_target_cross_entropy(input: Tensor, target_logits: Tensor, reduction: str = "mean") -> Tensor:
    """Soft-target cross-entropy −Σ p log q, with p = softmax(target_logits), q = softmax(input).

    The class dimension is 1 (0 for 1-D input), as in :func:`torch.nn.functional.cross_entropy`; ``reduction`` is
    ``"none"``, ``"mean"`` (over all non-class positions) or ``"sum"``. The student gradient q − p is computed from
    the logit differences in log space, so it is exactly 0 when the two are equal and stays accurate as they
    approach, whatever their relative offset. Same value as ``F.cross_entropy(input, F.softmax(target_logits,
    class_dim))``, except on classes masked to −inf in both, where this returns 0·log 0 := 0 instead of NaN.
    """
    return _soft_target(input, target_logits, False, reduction)


def kl_div_with_logits(input: Tensor, target_logits: Tensor, reduction: str = "mean") -> Tensor:
    """KL(p ‖ q), with p = softmax(target_logits), q = softmax(input), from logits on both sides.

    Class dimension as in :func:`soft_target_cross_entropy`. ``reduction`` is ``"none"`` (per position), ``"mean"``
    (over positions), ``"sum"``, or ``"batchmean"``: the sum divided by ``input.size(0)`` for inputs with a batch
    dimension, and by 1 for a single 1-D distribution (where :func:`torch.nn.functional.kl_div` would divide by the
    number of classes). Exactly 0 at equality, and small divergences are not rounding noise. Same value as
    ``F.kl_div(F.log_softmax(input, 1), F.softmax(target_logits, 1), reduction="none").sum(1)``, except on classes
    masked to −inf in both, where this returns finite values instead of NaN.
    """
    return _soft_target(input, target_logits, True, reduction)


# --------------------------------------------------------------------------------------------------- modules
class LogSoftmax(torch.nn.LogSoftmax):
    """Drop-in for :class:`torch.nn.LogSoftmax` using :func:`log_softmax`."""

    def forward(self, input: Tensor) -> Tensor:
        return log_softmax(input, self.dim, _stacklevel=5)


class Softmax(torch.nn.Softmax):
    """Drop-in for :class:`torch.nn.Softmax` using :func:`softmax`."""

    def forward(self, input: Tensor) -> Tensor:
        return softmax(input, self.dim, _stacklevel=5)


class Softmin(torch.nn.Softmin):
    """Drop-in for :class:`torch.nn.Softmin` using :func:`softmin`."""

    def forward(self, input: Tensor) -> Tensor:
        return softmin(input, self.dim, _stacklevel=5)


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Drop-in for :class:`torch.nn.CrossEntropyLoss` using :func:`cross_entropy`."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
