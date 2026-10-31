# softmaxfix

Softmax, log-softmax and cross-entropy for JAX and PyTorch that avoid **Softmax Collapse** (gradients that round to
zero at large logit margins) and **Softmax Cancellation** (soft-target gradients that do not vanish when student and
teacher agree), following Barron, Gnanapragasam & Tschannen, _A Fix for Softmax and Cross-Entropy_
([arXiv:2610.31642](https://arxiv.org/abs/2610.31642); HTML revision at
[jonbarron.info/softmaxfix](https://jonbarron.info/softmaxfix/)).

The functions are drop-in replacements with the native signatures:

```python
import softmaxfix.jax as sfx  # for jax.nn.log_softmax, jax.nn.softmax and the optax cross-entropies
import softmaxfix.torch as sft  # for F.log_softmax, F.softmax, F.softmin, F.cross_entropy and their nn modules

loss = sft.cross_entropy(logits, labels)  # same arguments as torch.nn.functional.cross_entropy
kl = sfx.kl_divergence_with_logits(student_logits, teacher_logits)  # teacher *logits*, not probabilities
```

Install from a checkout with the extra of the framework you use: `pip install ".[jax]"` or `pip install ".[torch]"`.

## Relation to the paper

The paper presents its fix with short reference listings for `log_softmax`, `softmax`, cross-entropy with integer
targets, and soft-target cross-entropy. This library implements the same ideas: the log-softmax of Eq. 5, which
never adds small exponentials to 1; cross-entropy centred at the target, as in Eq. 6; and the soft-target gradient
evaluated in log space from the logit difference, Eq. 9. It packages them as replacements for the native APIs.

| Paper listing                         | `softmaxfix.jax`                                                                                 | `softmaxfix.torch`                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| `log_softmax`                         | `log_softmax`                                                                                    | `log_softmax`, `LogSoftmax`                |
| `softmax`                             | `softmax`                                                                                        | `softmax`, `softmin`, `Softmax`, `Softmin` |
| `cross_entropy(z, t)`                 | `softmax_cross_entropy_with_integer_labels` (and `softmax_cross_entropy` for probability labels) | `cross_entropy`, `CrossEntropyLoss`        |
| `soft_target_cross_entropy(z_S, z_T)` | `soft_target_cross_entropy`                                                                      | `soft_target_cross_entropy`                |
| (not in the paper)                    | `kl_divergence_with_logits`                                                                      | `kl_div_with_logits`                       |

**The paper's rule is available as an option.** By default the library chooses, row by row, the exact rewriting of
each derivative that is the more accurate in floating point. With `rule="paper"` it uses the paper's derivative
rules in every row instead: those its listings get from automatic differentiation, under which the argmax entry of a
reverse-mode gradient is minus the sum of the others, so gradients sum to zero by construction.

```python
with sft.options(rule="paper"):  # or sfx.options(...); set_options(...) for the whole process
    loss = sft.cross_entropy(logits, labels)
```

**Why the listings are not called directly.** They are written to show the fix compactly, for the setting of the
paper: the last axis, unmasked inputs, first derivatives. A drop-in replacement has to cover more:

- arbitrary axes and masks, including rows where a single class is left;
- second and higher derivatives, for example the Hessians of a distillation loss;
- 16-bit inputs;
- every option of the native functions (class weights, label smoothing, `ignore_index`, probability targets).

So the library reimplements the ideas on its own kernels rather than wrapping the listings, and adds a few
numerical refinements:

- an exactly computed shift x − x_max;
- the per-row choice of derivative rule above;
- a variant for flush-to-zero arithmetic, `rule="flush_proof"`: the default in JAX, whose CPU backend flushes
  subnormals; torch, which does not by default, keeps `rule="accurate"`;
- the KL divergence from logits.

The module docstrings of `softmaxfix.jax` and `softmaxfix.torch` describe each one, with the cases it addresses.

**The listings are kept as the reference.** They are transcribed verbatim (JAX, `tests/paper_jax.py`) and ported
line by line (PyTorch, `tests/paper_torch.py`). The test suites compare the library with them case by case, on the
quantities the listings compute: values, the cross-entropy gradient and the sum of its entries, the softmax and
log_softmax VJPs, the soft-target loss and its gradients, and a KL divergence composed from the listings' functions
(CE − H; the listings have no KL). Each case is measured on one seed (four pooled
draws where rows are few). A check fails where the defaults or the paper's rule are less accurate than the paper's
code, in the worst case or on average beyond a margin, apart from a few documented exceptions, each diagnosed.

Twenty-seed runs of a comparison harness, used during development, measure more rows and more quantities.
`VALIDATION.md` summarises them, with the suites' method, the tolerance calibration, the costs and the paper's
experiments rerun with both implementations. They show what one seed per case cannot:

- a few worst cases beyond the suites' margin, in rows a single seed does not reach;
- the sums of the VJPs' entries. The paper's rules make these zero by construction. By default the library trades
  that for accuracy entry by entry, so in some regimes the sums sit further from zero than with the paper's code.
  With `rule="paper"` they are as close to zero as with the paper's code on average, apart from a few cases in JAX,
  mostly 16-bit, listed in `VALIDATION.md`.

## Tests

```bash
pip install -e ".[jax,torch,test]"
pytest                                     # both suites; GPU cases run where a device is present
pytest -m "not slow"                       # without the torch.compile test
pytest tests/test_torch.py --seed 101      # another master seed for the generated inputs (default 20261026)
```

Accuracy is graded against an mpmath oracle, with tolerances calibrated from the library's own measured errors
(`tests/tolerances_*.json`, merged by `tests/calibrate.py` from the raw files in `tests/calibration/`). The tolerances
were calibrated on a development build of jax (0.12.1.dev20261026) and on torch 2.14.1; other builds may round
differently in places. `VALIDATION.md` describes the suites and the calibration.
