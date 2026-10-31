"""The paper's reference implementation (HTML revision, https://jonbarron.info/softmaxfix/), verbatim, as a comparator.

The comparator of the accuracy tests (softmaxfix.jax must never be less accurate than the code it departs from) and the
paper-code arm of the experiments in VALIDATION.md. Last axis only, as in the paper. ``kl_divergence`` is not in the
paper: it is KL(p ‖ q) = CE(p, q) − H(p) composed from the paper's own functions, the direct way to obtain a KL from its
code, as the comparator for softmaxfix's KL.
"""

import jax
import jax.numpy as jnp


def log_softmax(z):
    i_max = jnp.argmax(z, axis=-1)
    z_max = jnp.take_along_axis(z, i_max[..., None], axis=-1)  # Max logit.
    is_max = jax.nn.one_hot(i_max, z.shape[-1], dtype=bool)  # Max mask.
    shifted = jnp.where(is_max, 0.0, z - z_max)  # Exact 0 at argmax.
    others = jnp.where(is_max, -jnp.inf, shifted)  # Exclude argmax from sum.
    m = -jax.nn.logsumexp(others, axis=-1, keepdims=True)  # Logit margin.
    return shifted - jax.nn.softplus(-m)  # log_softmax(z).


def softmax(z):
    return jnp.exp(log_softmax(z))


def cross_entropy(z, t):
    z_t = jnp.take_along_axis(z, t[..., None], axis=-1)  # Target logit.
    is_t = jax.nn.one_hot(t, z.shape[-1], dtype=bool)  # Target mask.
    others = jnp.where(is_t, -jnp.inf, z - z_t)  # Exclude target from sum.
    m = -jax.nn.logsumexp(others, axis=-1)  # Logit margin.
    return jax.nn.softplus(-m)  # -log_softmax(z)_t.


def soft_target_cross_entropy(z_S, z_T):
    log_q = log_softmax(z_S)
    q, p = jnp.exp(log_q), softmax(z_T)
    delta = z_S - z_T  # Logit difference.
    d = delta - jnp.sum(p * delta, axis=-1, keepdims=True)  # Center under p.
    s = d - jnp.log1p(jnp.sum(p * jnp.expm1(d), axis=-1, keepdims=True))  # log(q / p).
    u = jnp.where(jnp.abs(s) < 1, p * jnp.expm1(s), q - p)  # Difference q - p.
    g = u - q * jnp.sum(u, axis=-1, keepdims=True)  # Ensure sum(g) = 0.
    sg = jax.lax.stop_gradient
    loss = -jnp.sum(p * sg(log_q), axis=-1)  # Forward loss.
    return loss + jnp.sum(sg(g) * (z_S - sg(z_S)), axis=-1)  # Custom gradient g.


def kl_divergence(z_S, z_T):  # Not in the paper (see the module docstring).
    log_p = log_softmax(z_T)
    return soft_target_cross_entropy(z_S, z_T) + jnp.sum(jnp.exp(log_p) * log_p, axis=-1)
