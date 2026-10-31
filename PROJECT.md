# softmaxfix — project context

JAX and PyTorch implementations of Barron, Gnanapragasam & Tschannen, _A Fix for Softmax and Cross-Entropy_
(arXiv:2610.31642): softmax, log-softmax and cross-entropy without Softmax Collapse or Softmax Cancellation, as drop-in
replacements for `jax.nn` / `optax` and `torch.nn.functional` / `torch.nn`, plus soft-target cross-entropy and KL
divergence from teacher logits. The library is meant for work where numerical accuracy is paramount.

The reference text is the paper's HTML revision (https://jonbarron.info/softmaxfix/). It supersedes PDF v1, whose
soft-target listing lacks the |s| < 1 fallback (NaN gradients on −inf masks and extreme teacher gaps).

## Layout

- `src/softmaxfix/__init__.py` — the version only (hatch-vcs, with a source-checkout fallback); imports no framework.
- `src/softmaxfix/jax.py`, `src/softmaxfix/torch.py` — the two back-ends, independent of each other. Their module
  docstrings state every difference from the paper's listings, with its measured effect, and the numerical options.
- `tests/oracle.py` — the mpmath reference: closed forms on the exact binary inputs, never the paper's formulas or the
  library's; self-tested by `tests/test_oracle.py`.
- `tests/cases.py` — the input regimes. `tests/test_jax.py`, `tests/test_torch.py` — the suites.
- `tests/paper_jax.py` (verbatim) and `tests/paper_torch.py` (line-by-line port; `naive_*` without the float32 upcast)
  — the paper's listings, as references for the accuracy tests only.
- `tests/calibrate.py`, `tests/tolerances.py`, `tests/tolerances_{jax,torch}.json`, `tests/calibration/` — tolerance
  calibration (below).
- `VALIDATION.md` — how the library was validated, and the evidence behind every figure quoted in the docstrings and
  the README. The development harnesses, probes, benchmarks and full results log are kept outside the repository.

## Domain terms

- **Softmax Collapse:** gradients that round to zero at large logit margins (the target's probability rounds to 1).
- **Softmax Cancellation:** soft-target gradients that do not vanish when student and teacher agree.
- **Options** `Options(exact_shift, rule)`, set with `set_options(...)` or `with options(...):`.
    - `exact_shift`: x − x_max kept exactly (TwoSum) instead of rounded.
    - `rule="accurate"`: per row, the exact rewriting of each derivative that is accurate for it; torch's default.
    - `rule="flush_proof"`: `"accurate"` plus a scaled branch for rows whose products would flush to zero; JAX's
      default, because XLA CPU flushes subnormals.
    - `rule="paper"`: the paper's derivative rule (zero-sum by construction) on this library's kernels — the paper's
      _rule_, not its _code_.
- **Native:** `jax.nn` / `optax` / `torch.nn.functional`, the code the paper fixes. In the suites it is the known-bad
  case (must fail where the bugs bite) and a baseline to beat; never a known-good reference.
- **The paper's code:** the listings in `tests/paper_*.py`.
- **Regimes:** hard targets `safe`, `wide`, `flat`, `cliff`, `margin`, `margin_bigk` (pooled over 4 draws, 90th
  percentile as the worst case), `wrong`; soft targets `teacher`, `offset`, `outlier`, `pairs`.
- **Units:** ulp of the format (componentwise) or ε relative to the row's largest component (normwise); a flush-to-zero
  allowance (`oracle.flushable`) covers mass that flushing subnormals can remove. **Σg:** the zero-sum residual of a
  gradient.

## Conventions

- **Accuracy first.** A change that is better than the paper is adopted only once measured against every alternative
  (the shipped code, the paper's code, native, other candidates), on worst cases and componentwise metrics as well as
  means. A deviation from the paper's listings must be shown, by tests and by a rounding-error argument, not to be less
  accurate than them; where it is less accurate somewhere, both paths stay selectable. Defaults may differ per framework
  when measured and documented.
- **Tolerances come from measurement, never by hand.** `tests/calibrate.py measure` then `merge` sets each tolerance to
  the next power of two at or above 2 × the library's worst error over the calibration seeds (floor 2); native's errors
  never set one. Validate on held-out seeds 101–110. The merge refuses raw files of mixed devices, options or library
  source (SHA-256): a recalibration re-measures every seed (`measure --force`).
- **Tests must be passable.** A test that can fail by chance against a broken baseline is removed or given a
  deliberate, measured exclusion with its numbers in a comment (as `NATIVE_COMPARISON_EXCLUDED`). Loosening a tolerance,
  adding an exclusion or changing what a test checks is the maintainer's decision, backed by a pooled re-measurement.
- **Figures are traceable.** Every number in a docstring or the README appears in `VALIDATION.md` with its comparands,
  framework, device, seeds and library version (eager or `jit`, where it matters).
- **Tone towards the paper.** In user-facing text the differences are additions for cases the listings do not target
  (masks, higher derivatives, 16-bit inputs, native API options), stated with their measured effect, not "defects".
- **Style.** Ruff with the vendored `ruff.toml` (line length 120); British English in prose.
- **Publishing.** A push to `main` runs `build.yml`, which uploads the wheel to the maintainer's GemFury index; a `v*`
  tag also creates a GitHub release. The test workflow runs on manual dispatch only: the CPU suites take tens of
  minutes, and the tolerances were calibrated on a development build of jax and on torch 2.14.1.
