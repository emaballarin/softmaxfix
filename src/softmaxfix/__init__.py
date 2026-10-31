"""Softmax, log-softmax and cross-entropy without Softmax Collapse or Softmax Cancellation.

Implements Barron, Gnanapragasam & Tschannen, "A Fix for Softmax and Cross-Entropy" (arXiv:2610.31642,
HTML revision at https://jonbarron.info/softmaxfix/). Framework back-ends live in ``softmaxfix.jax`` and
``softmaxfix.torch``; this module imports neither.
"""

try:  # written at build time by hatch-vcs, absent in a plain source checkout
    from softmaxfix._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"
