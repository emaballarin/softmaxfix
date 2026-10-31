"""The paper's reference implementation (HTML revision), ported line by line to PyTorch, as a comparator.

The comparator of the accuracy tests (softmaxfix.torch must never be less accurate than the code it departs from) and
the paper-code arm of the experiments in VALIDATION.md. ``jax.nn.softplus`` is ``logaddexp(x, 0)``; torch's
``F.softplus`` switches to the identity above 20, which would make the port less accurate than the paper, so
``torch.logaddexp`` is used. Half-precision inputs are computed in float32, as in softmaxfix.torch, so that only the
algorithms differ; the ``naive_*`` functions are the same port without that upcast, computing in the input dtype exactly
as written. Last axis only, as in the paper. ``kl_divergence`` is not in the paper: it is KL(p ‖ q) = CE(p, q) − H(p)
composed from the paper's own functions, as in ``paper_jax``.
"""

import math

import torch
import torch.nn.functional as F


def _softplus(x):
    return torch.logaddexp(x, torch.zeros_like(x))


def _upcast(fn):
    def wrapped(*args):
        dtype = args[0].dtype
        if dtype in (torch.float16, torch.bfloat16):
            args = tuple(a.float() if a.is_floating_point() else a for a in args)
            return fn(*args).to(dtype)
        return fn(*args)

    return wrapped


def _log_softmax(z):
    i_max = z.argmax(-1)
    z_max = z.gather(-1, i_max[..., None])  # Max logit.
    is_max = F.one_hot(i_max, z.shape[-1]).bool()  # Max mask.
    shifted = torch.where(is_max, 0.0, z - z_max)  # Exact 0 at argmax.
    others = torch.where(is_max, -math.inf, shifted)  # Exclude argmax from sum.
    m = -torch.logsumexp(others, -1, keepdim=True)  # Logit margin.
    return shifted - _softplus(-m)  # log_softmax(z).


def _cross_entropy(z, t):
    z_t = z.gather(-1, t[..., None])  # Target logit.
    is_t = F.one_hot(t, z.shape[-1]).bool()  # Target mask.
    others = torch.where(is_t, -math.inf, z - z_t)  # Exclude target from sum.
    m = -torch.logsumexp(others, -1)  # Logit margin.
    return _softplus(-m)  # -log_softmax(z)_t.


def _soft_target_cross_entropy(z_S, z_T):
    log_q = _log_softmax(z_S)
    q, p = log_q.exp(), _log_softmax(z_T).exp()
    delta = z_S - z_T  # Logit difference.
    d = delta - (p * delta).sum(-1, keepdim=True)  # Center under p.
    s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))  # log(q / p).
    u = torch.where(s.abs() < 1, p * torch.expm1(s), q - p)  # Difference q - p.
    g = u - q * u.sum(-1, keepdim=True)  # Ensure sum(g) = 0.
    loss = -(p * log_q.detach()).sum(-1)  # Forward loss.
    return loss + (g.detach() * (z_S - z_S.detach())).sum(-1)  # Custom gradient g.


def _kl_divergence(z_S, z_T):  # Not in the paper (see the module docstring).
    log_p = _log_softmax(z_T)
    return _soft_target_cross_entropy(z_S, z_T) + (log_p.exp() * log_p).sum(-1)


naive_log_softmax = _log_softmax
naive_kl_divergence = _kl_divergence
naive_softmax = lambda z: _log_softmax(z).exp()
naive_cross_entropy = _cross_entropy
naive_soft_target_cross_entropy = _soft_target_cross_entropy
log_softmax = _upcast(_log_softmax)
softmax = _upcast(lambda z: _log_softmax(z).exp())
cross_entropy = _upcast(_cross_entropy)
soft_target_cross_entropy = _upcast(_soft_target_cross_entropy)
kl_divergence = _upcast(_kl_divergence)
