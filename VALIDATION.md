# Validation

This file records how the library was validated: the evidence behind the figures quoted in the module docstrings and the README. The measurements were made from 26 to 31 October 2026 during development. Besides the test suites, they used comparison harnesses, benchmarks and probe scripts. Those tools, their raw outputs and the full development log are not part of this repository; the figures below come from them. What can be rerun here: the test suites, the calibration script (`tests/calibrate.py`), the tolerance files and the raw calibration data they were merged from (`tests/calibration/`).

Units: errors are in ulps of the format for componentwise metrics, and in ε relative to the row's largest component (normwise) otherwise. "fp16", "bf16", "fp32" and "fp64" are the input formats. "worst / mean" pairs give both.

## Machines and versions

| Machine             | Hardware                                  | Software                                                                                         |
| ------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Laptop              | RTX 3050 Ti Laptop GPU (4 GB), laptop CPU | Python 3.14.10, numpy 2.5.4, jax 0.12.1.dev20261026, optax 0.2.8, torch 2.16.0.dev20261027+cu134 |
| Server (`iitcsml2`) | GTX 1080 Ti GPUs, multi-core CPU          | Python 3.14.9, numpy 2.5.4, jax 0.12.1.dev20261026, optax 0.2.8, torch 2.14.1+cu126 (CUDA 12.6)  |

- Until 27/10/2026 the laptop ran jax 0.12.0 and an earlier torch nightly. The figures dated 26/10/2026 below were measured there.
- The tolerance files were calibrated on the server: a development build of jax, and torch's 2.14.1 release. Other builds may round differently in places; the suites have not been run against current release wheels.

## How the suites test accuracy

**Reference.** `tests/oracle.py` evaluates closed forms in mpmath on the exact binary inputs.

- Precision is set per row: 60 guard digits, plus enough to resolve e^{−span}. A fixed 80 digits returned 0 for a log-probability of −2.5·10⁻⁸³ at margin 192.
- Teacher gradients use exact logit identities.
- `tests/test_oracle.py` checks these, and every derivative formula, against 80-digit numerical differentiation.
- The oracle's formulas are analytic; it uses neither the paper's formulas nor this library's.

**Flush-to-zero allowance.** XLA's CPU backend flushes subnormal operands and results to zero, and so does torch on CPU under `torch.set_flush_denormal(True)`; the GPUs measured here did not. The metrics allow for this:

- every metric allows for the mass that flushing the row's subnormal operands can remove (`oracle.flushable`);
- derivative outputs also allow for the reference's own subnormal entries (`oracle.subnormal_mass`) and for float16's gradual-underflow rounding (`oracle.underflow_rounding`).

For 16-bit formats, which are computed in float32, the normal threshold used is float32's (`oracle.FLUSH_EXPONENT`). With float16's own threshold, the allowance forgave about 62 ε normwise on fp16 `flat`.

**Regimes** (`tests/cases.py`):

- Hard targets:
    - `safe`: N(0, 1) logits.
    - `wide`: scales 10 and 100, offset 10³.
    - `flat`: 2 048 near-uniform classes, target ≠ argmax.
    - `cliff`: margins 0–40, or 0–24 in 16-bit formats.
    - `margin`: margins up to underflow.
    - `margin_bigk`: 4 096 classes.
    - `wrong`: target up to 60 below the argmax.
- Soft targets:
    - `teacher`: gaps from 10⁻¹ down to 10⁻¹².
    - `offset`: near the teacher, plus a common offset c ∈ {4, 16, 64}.
    - `outlier`: one class 40–90 below the teacher's others.
    - `pairs`: independent logits.
- Also: probability-label cross-entropy on smoothed one-hot labels; Hessians in four regimes; hand-made edge rows (−inf masks, K = 1, ties, non-finite inputs).

**Tolerances.** `tests/calibrate.py measure` writes one raw file per seed. `merge` sets each (framework-device, regime, metric, format) tolerance to the next power of two at or above twice the library's worst error, with a floor of 2. The merge refuses to write in any of these cases:

- a library error is non-finite or above 2 048;
- a must-fail native error (below) falls within its tolerance;
- the raw files differ in devices, library options or library source (SHA-256).

Native's errors never set a tolerance. In the regimes where native is usually accurate, they are recorded and only reported.

**Native as the known-bad case.** On `cliff`, `teacher` and `offset`, native must exceed the tolerance; otherwise the test fails, and so does the calibration. This shows the tests can detect the two bugs.

There is one exclusion: torch on CUDA, fp16 `cliff`, the whole regime (decided 29/10/2026). Native torch computes fp16 in float32 on CUDA, so its cliff (margin 16) lies beyond the bottom of fp16's normal range (margin 9.7). Over the calibration seeds 1–10 (the raw files in `tests/calibration/torch/`), native's smallest error divided by the tolerance there is:

- 0.67 for the value and log_softmax;
- 0.91 for Σg;
- 2.38 for the gradients (thin);
- 1.0·10³ for the argmax VJP.

**Comparisons with native and with the paper's code.** Two families of tests compare per-regime aggregates, not single rows: `test_not_less_accurate_than_native` and `test_not_less_accurate_than_the_paper` (including the non-default option sets). Native is the baseline to beat. The paper's listings are the code this library departs from.

- **When a regression is flagged.** The library is flagged if its worst error exceeds 1.25 × the reference's + 4, or its mean exceeds 1.25 × the reference's + 1.
- **The pooled regime, `margin_bigk`.** It has 8 rows per draw, too few for a stable per-seed comparison: under the plain rule, known-good code failed on 1–2 seeds in 20. So the test pools 4 draws there. The worst case becomes the 90th percentile of the rows, and any extra non-finite row fails outright.
- **Calibration of that rule.** On pooled data (30 seeds × 8 draws, pools of 4, both frameworks, CPU and GPU), known-good comparisons failed:
    - 135 of 4 800 pools under the maximum;
    - 20 of 4 800 under the 90th percentile.

    Known regressions failed 1 200 of 1 440 pools under both rules. The 240 that pass are torch's port of the paper's code in 16-bit, which is not worse there.

    Those data predate `rule="flush_proof"`. JAX's current default, the explicit `accurate` option sets and torch's `flush_proof` sets are not in them.

**The deliberate exclusion from the comparison with native.** The JAX comparison with native skips `log_softmax_vjp` in `margin_bigk` on the GPU, in float32 and float64 (`NATIVE_COMPARISON_EXCLUDED`). There the library ties native, and the per-seed rule fires by chance. Measured over base seeds 1001–1060, pooled as the test pools:

| GPU, `log_softmax_vjp` | Rule fires | Mean, ours / native (paired difference, 95 % CI) | q90         | q99         | max         |
| ---------------------- | ---------- | ------------------------------------------------ | ----------- | ----------- | ----------- |
| float32                | 3 of 60    | 2.24 / 2.23 (+0.009, [−0.16, +0.18])             | 5.56 / 4.97 | 19.4 / 21.0 | 39.7 / 42.9 |
| float64                | 1 of 60    | 2.08 / 2.14 (−0.063, [−0.22, +0.10])             | 5.00 / 5.15 | 17.5 / 19.9 | 40.8 / 34.1 |

- On the CPU the rule fires on none of the 60 seeds, in either format.
- In float16 and bfloat16 no metric is flagged on any of the 60 seeds, on either device, and the library's mean is below native's on every metric.
- The exclusion was added after held-out seed 105 failed this test, in float32 (879 of 880 cases passed).
- Against the paper's code, the same key was traced to summation order: on the GPU, XLA reduces the 4 095-term argmax sum in an order that differs between implementations and between runs. Beating it would need a compensated sum inside the transpose of a `jax.custom_jvp`, and JAX 0.12 exposes no custom transpose.

**Registered expected failures.** In `test_options_not_less_accurate_than_the_paper` (`OPTION_DEVIATIONS`), a known deviation xfails only when the comparison flags exactly its metrics; any other flagged metric fails.

- **JAX, the rounded-shift option sets, `wide` float64 `ce_int_value`.** This is the rounded shift's own error: one row of 192 above the paper's worst (15.2 against 8.2 ulp), with a better mean (0.37 against 0.48 on CPU; 0.38 against 0.46 on GPU). The default exact shift removes it.
- **torch, the paper-rule option sets, `flat` float64 `softmax_vjp`.** The paper rule's pivot cancels on flat rows, in ours and in the paper's code alike.
    - Pooled over 4 seeds, ours is the more accurate (worst 44.2 against 55.7 ε, mean 13.1 against 14.9 ε).
    - Over 20 seeds the paper rule is better than the paper's code there: CPU mean −6.20 ε (95 % CI [−8.72, −3.88]) and worst 51.5 against 98.5 ε; CUDA −5.76 ε and 42.2 against 81.3 ε.
    - On the server the xfail does not trigger.

**Other checks.**

- **API parity with native, on every option.** This covers:
    - optax's `axis`, `where` and label wrapping;
    - a 108-configuration `cross_entropy` grid (class weights, `ignore_index`, label smoothing, probability targets, reductions, 1-D/2-D/4-D inputs);
    - legacy arguments, the module wrappers, `dim=None` warnings, non-contiguous and empty inputs, non-finite inputs.
- **Transforms.**
    - JAX: `check_grads` to order 3 in forward and reverse mode, `jit`, `vmap`, and `jacfwd == jacrev`.
    - torch: `gradcheck`, `gradgradcheck` and forward AD; `torch.func` `grad`, `vmap`, `jacrev`, `jacfwd` and `hessian`; and a `torch.compile` smoke test.
- **Precision policy.** Half-precision values are bitwise the float32 computation rounded once. In torch, a mutant computing half precision in half precision fails this test.

## Calibration and final validation

| Tolerance file                | Calibration seeds | Devices   | Options                                  | Library source (SHA-256)         | Date       |
| ----------------------------- | ----------------- | --------- | ---------------------------------------- | -------------------------------- | ---------- |
| `tests/tolerances_jax.json`   | 1–30              | CPU, GPU  | `exact_shift=True`, `rule="flush_proof"` | `b800759a84a49d7a…` (`jax.py`)   | 30/10/2026 |
| `tests/tolerances_torch.json` | 1–10              | CPU, CUDA | `exact_shift=True`, `rule="accurate"`    | `65c6442405fbf13a…` (`torch.py`) | 30/10/2026 |

**Source hashes.** The hashes are those of the modules as measured, during development.

- The shipped modules differ from them only in documentation (docstrings and comments): once docstrings are removed, their abstract syntax trees are identical. This was checked against the development copies, which are not part of this repository.
- The calibration's merge refuses raw files of different sources. So seeds measured with the shipped modules cannot be added to the shipped raw files: a recalibration re-measures every seed (`tests/calibrate.py measure --force`).

**Held-out validation.** On seeds 101–110, every tolerance-dependent test passed on the server, CPU and GPU:

- the library's accuracy, 880 / 880 per framework;
- must-fail: JAX 240 / 240, torch 230 / 230 (plus 10 skipped, the CUDA fp16 `cliff` exclusion);
- Hessians, 80 / 80;
- edge rows, 140 / 140;
- the comparisons with native: torch 880 / 880; JAX 879 / 880, then 880 / 880 rerun with the exclusion above.

These seeds are not unseen in every respect: the rerun followed the exclusion, and the `safe` regime's rows at these seeds had been probed before (native's tails).

**Native, reported only.** In the regimes where native is usually accurate, native exceeds the library-alone tolerance on 100 of 240 keys (JAX) and 71 of 240 (torch).

- On the JAX CPU keys (seeds 1–30), native's worst would have raised the tolerance on 83 of 120 keys, by a median 4× (up to 64×); over all 120 keys the median is 2×.
- The held-out cases examined were single rows with cancellation in native's formulas, for example optax's (m + lse) − z_t for the integer-label value.

**Final suites** (the default seed; every suite in `tests/`):

| Target                                 | torch (with the oracle's self-tests) | JAX                   |
| -------------------------------------- | ------------------------------------ | --------------------- |
| Laptop CPU (30/10/2026)                | 521 passed, 2 xfailed                | 501 passed, 3 xfailed |
| Server, CPU and GPU (CUDA), 31/10/2026 | 987 passed, 1 skipped                | 874 passed, 6 xfailed |

## Differences from the paper's reference listings

The paper's listings show the fix compactly, for its setting: the last axis, unmasked inputs and first derivatives. A drop-in replacement must also cover the cases below. The listings are kept verbatim (`tests/paper_jax.py`) and ported line by line (`tests/paper_torch.py`) as the comparator of the accuracy tests.

**How to read the "Measured" column.** Each figure says what it compares. Unless stated otherwise, it gives worst / mean over seeds 1–5 of the suites' regimes, JAX CPU, measured 26/10/2026. That was the library as it then was: before the exact shift, the centred softmax rule and the float32 computation of 16-bit inputs. The twenty-seed comparisons below supersede them where they overlap.

| Listing                                       | Case the listing does not target                                                                                       | This library                                                                                                                        | Measured                                                                                                                                                                                                                                                                            |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| All: the shift x − x_max                      | Its rounding, amplified by exp into a relative error of about ε·\|x_j − x_max\|/2 (shared by native and the listings)  | Exact shift: hi + lo by TwoSum, e^{hi} + e^{hi}·lo (option `exact_shift`)                                                           | This library, rounded → exact shift (20 seeds, server), fp64 `wide`, torch CPU / JAX CPU / JAX GPU: softmax 306 → 3.7 / 4.5 / 3.1 ulp, CE gradient 306 → 4.3 / 4.7 / 3.5 ulp, CE Hessian 74 → 3.6 / 2.7 / 2.8 ε; fp32 `margin_bigk` softmax, torch CPU 64 → 6.7 ulp, torch CUDA 65 → 4.1 ulp (numpy probe: 64.7 → 4.7)                                                                                |
| `log_softmax` (Eq. 5), derivative by autodiff | Flat rows whose target is not the argmax: the argmax entry, −Σ of the others, cancels                                  | Per-row derivative rule (`rule="accurate"`); the paper's rule is `rule="paper"`                                                     | Autodiff of Eq. 5, argmax gradient, fp32: 6.9·10⁴ ε at 32 000 classes (native about 1 ε). log_softmax VJP on `flat`, 20 seeds: the paper's code 70–118 ε; the paper's rule 47–49 ε (fp32) and 37–44 ε (fp64) in the variants harness (laptop), and `rule="paper"` itself 46–49 ε (fp32) and 37–50 ε (fp64) on the server; `rule="accurate"` 0.47 and 0.44 ε                            |
| `log_softmax` value                           | `shifted − softplus(logsumexp(others))`: a log–exp round trip                                                          | `shifted − log1p(Σ_{j≠max} e^{x_j−x_max})`, the same Eq. 5 quantity                                                                 | The paper's code → this library: fp64 `margin` 640 / 97 → 305 / 52 ulp                                                                                                                                                                                                              |
| `softmax = exp(log_softmax)`                  | The value carries the rounding of log q, ε·\|log q\|                                                                   | Value e^{x−x_max}/Σ; derivative from the library's rules                                                                            | The paper's code → this library: fp32 `flat` 14.5 / 8.52 → 2.98 / 2.04 ulp                                                                                                                                                                                                          |
| softmax derivative                            | Large, nearly equal cotangent entries on confident rows; XLA CPU flushing the subnormal products the argmax entry sums | Cotangent centred at the argmax on confident rows; `rule="flush_proof"` forms the products at scale                                 | Numpy probe, fp32 `wide`: 151 ε (the paper's rule) → 1.8 ε (centred). JAX CPU fp32 `margin_bigk`, an uncentred pivot under flushing: 5.4·10⁴ ε for the paper's code evaluated eagerly (20 seeds, laptop) and for an uncentred pivot on confident rows under `jit` (server); jitted, the paper's code reads 18.8 ε there, `rule="paper"` 2.9 ε and `rule="accurate"` 5.3 ε. The flush band: below                                                                                              |
| Hard-target CE, Eq. 6                         | A target far below the argmax: z_j − z_t is rounded at the size of the target's deficit                                | Value from `log_softmax`; gradient q_j off the target and −Σ_{j≠t} q_j on it                                                        | Eq. 6, fp16, deficit 262: 7 % gradient error (0.377 against 0.407). The paper's code → this library: fp64 `margin_bigk` value 353 / 77 → 11.7 / 2.4 ulp                                                                                                                             |
| logsumexp over an all −inf set                | One class left unmasked: NaN gradient                                                                                  | Zero gradient, as native                                                                                                            | Edge-row tests                                                                                                                                                                                                                                                                      |
| Soft-target CE, Eq. 9 on δ = z_S − z_T        | A common logit offset c: δ is rounded at the size of c                                                                 | δ with its exact remainder (TwoSum), measured from the teacher's mode                                                               | The paper's code → this library: `offset` fp32 student gradient 1.9·10⁷ / 1.4·10⁶ → 2.5 / 0.66 ε. Eq. 9 on rounded δ, in this library's first implementation: 1.05·10⁵ ε at c = 64 (gap 10⁻³)                                                                                       |
| Soft-target CE, \|s\| < 1 fallback            | Far classes' expm1 rounding dominating the normaliser; an underflowed q                                                | Close set \|b\| ≤ 1 in log space; far classes from their own probabilities; normaliser corrected by log(1 − Q_far) − log(1 − P_far) | This library's first implementation, every class in the sum: bf16 KL −30 %                                                                                                                                                                                                          |
| Soft-target CE, supports                      | Classes the teacher gives zero probability (masked, truncated, underflowed teachers)                                   | q rescaled on the teacher's support, in log space; q − p only where the loss is +inf                                                | The per-element patch gives [−0.25, 0.25] instead of [−0.5, 0.5] for z_T = [0, −120], z_S = [0, 0]                                                                                                                                                                                  |
| Soft-target CE, `stop_gradient`               | Second derivatives with respect to the student                                                                         | Differentiable derivative rules (`jax.custom_jvp`, `torch.autograd.Function`)                                                       | Hessian against diag(q) − qqᵀ: within 2.8·10⁻¹⁷ (fp64); the listing's is 0                                                                                                                                                                                                          |
| Soft-target CE, masked in both                | −inf − (−inf) in the value: NaN                                                                                        | Classes masked in both contribute exactly 0                                                                                         | Edge-row tests                                                                                                                                                                                                                                                                      |
| Soft-target CE, teacher gradients             | Autodiff through p of −Σ p log q carries the logsumexp's rounding                                                      | Logit identities: −p ⊙ (z_S − ⟨p, z_S⟩) for CE, −p ⊙ (b − ⟨p, b⟩) for KL                                                            | The paper's code → this library: fp32 `pairs` 76 / 3.1 → 4.2 / 0.95 ε (native 1.2·10⁴); fp32 `teacher` 20 / 1.7 → 3.4 / 0.87 ε                                                                                                                                                      |
| (no KL listing)                               | KL composed as CE − H(p) cancels when KL ≪ CE, the regime distillation lives in                                        | KL in log space: S′ + log1pmx(S′ + r), far terms p_k(r* − b_k)                                                                      | CE − H from the listings → this library: value, fp32 `teacher`, 2.3·10¹⁵ → 3.7 ulp (table below). This library's relative error ≤ 6 ε at every gap from 10⁻¹ to 10⁻¹²                                                                                                               |
| 16-bit inputs                                 | Every per-row scalar rounded to fp16 or bf16                                                                           | Computed in float32 and rounded once, in both frameworks                                                                            | JAX, this library's earlier input-dtype policy → float32 (27/10/2026): `pairs` bf16 KL value 7.15 → 0.50, `teacher` bf16 KL student gradient 4.69 → 0.46. Before the change, on the `pairs` KL value, the paper's code (fp16: 4.05 against 5.1) and native (bf16: 5.3 against 7.15) were ahead |

**The KL comparison** (worst error per implementation, seeds 1–5; JAX CPU / torch CUDA on the laptop; run 27/10/2026, on the library before the exact shift):

| KL, worst error                 | Ours      | CE − H from the listings | Native          |
| ------------------------------- | --------- | ------------------------ | --------------- |
| value, `teacher` fp32           | 3.7 / 3.4 | 2.3e15 / 3.6e16          | 1.3e38 / 8.5e37 |
| value, `offset` fp32            | 4.1 / 4.0 | 7.0e13 / 7.0e13          | 6.6e13 / 7.6e13 |
| value, `pairs` fp32             | 5.9 / 4.9 | 6.7 / 15.6               | 22.8 / 32.6     |
| student gradient, `offset` fp32 | 2.5 / 2.6 | 1.9e7 / 1.7e7            | 2.7e7 / 1.6e8   |
| teacher gradient, `pairs` fp32  | 4.1 / 4.6 | 28.5 / 33.1              | 1.3e4 / 1 340   |

The listings target cross-entropy, not KL; this table records why the KL needed its own algorithm.

**The torch port of the listings.** Over 20 seeds it gives non-finite cross-entropy Hessians on some `wide` and `margin` rows (31–44 of 3 840 and 60–140 of 960). The JAX transcription has none. The suspected cause is torch's double backward through the port's logsumexp at underflow; it was not investigated.

## Twenty-seed comparisons

The suites compare one seed per case. Two comparison harnesses measured more, over seeds 1–20:

- **The candidates harness** (server; JAX CPU, JAX GPU, torch CPU and torch CUDA).
    - Per-row errors against the oracle for each implementation, on the seven hard-target regimes plus `flush_bigk`: rows at 4 096 classes whose non-argmax probabilities straddle the bottom of the normal range, from −8 to +6 around it.
    - Hard-label quantities only: values, the cross-entropy gradient and Hessian, the log_softmax and softmax VJPs and JVPs, and their sums.
    - Extra metrics: componentwise and "tight" variants (the flush allowance without its sum term), and separately compiled VJPs.
    - Soft targets and KL were compared in the suites (one seed per case) and over seeds 1–5 by a reference-comparison script (laptop, 26–27/10/2026), not at 20 seeds.
- **The variants harness** (laptop): derivative-rule variants of log_softmax and softmax, and the correctly rounded gradient's own Σg.

**Comparison rule, per key** (regime, format, metric). A is worse than B when any of these holds:

- it has more non-finite rows;
- its paired mean difference exceeds 0.05 with the 95 % interval above 0;
- its worst exceeds 1.5 × B's worst + 0.5.

"Better" is the mirror image. A key where A is both worse and better is "mixed".

**Defaults against this library's previous rules and against the paper's code** (candidates harness, 356 keys per target, 27/10/2026). The previous rules are the rounded shift and the softmax VJP pivoted with r·v_max.

| Target     | Better than previous | Worse | Mixed | Against the paper's code: better / worse / mixed |
| ---------- | -------------------- | ----- | ----- | ------------------------------------------------ |
| JAX CPU    | 94                   | 0     | 1     | 254 / 25 / 2                                     |
| JAX GPU    | 89                   | 3     | 1     | 240 / 31 / 2                                     |
| torch CPU  | 87                   | 3     | 1     | 120 / 23 / 3                                     |
| torch CUDA | 88                   | 3     | 2     | 120 / 24 / 2                                     |

- The JAX rows are the exact shift with `rule="accurate"`, JAX's default until 29/10/2026.
- Against the previous rules, every flag is a single-row worst case except two mean shifts on fp64 `margin_bigk` log_softmax Σg: +0.08 ε on JAX GPU (with the same top rows), +0.10 ε on torch CPU.
- Against the paper's code, the JAX losses are all on the zero-sum residual Σg except one mixed key: JAX CPU fp64 `safe` softmax JVP, worst 8.36 against 4.05 ε, with the better mean. The torch losses are mostly Σg.

**`rule="flush_proof"`, JAX's default**, against `rule="accurate"` (candidates harness, 29–30/10/2026):

- **Off the flush band:** it ties on every key, 664 per device on CPU and GPU.
- **On the flush-band rows (XLA CPU), in the tight metrics:**

| `flush_bigk`, worst (ε)  | `rule="accurate"` | the paper's code | `rule="flush_proof"` |
| ------------------------ | ----------------- | ---------------- | -------------------- |
| softmax VJP, fp32 / fp64 | 2 088 / 8.3·10¹¹  | 4 089 / 1.6·10¹² | 0 / 0                |
| softmax JVP, fp32 / fp64 | 426 / 2.3·10¹¹    | 29.1 / 0         | 0 / 0                |

- **The paper's code does better in forward mode.** Its forward rule sums nothing at the argmax, so in forward mode on these rows `rule="accurate"` is less accurate than the paper's code.
    - On seeds 1–3, quoted in `softmaxfix.jax`'s docstring, the figures were 263 ε and 2.3·10¹¹ ε against 17.3 ε and 0.
    - The fp64 figure is one row (seed 1, row 2): its argmax entry, −5.738·10⁻³⁰⁴, came out as −5.737·10⁻³⁰⁴, so 1.7·10⁻⁴ of it was lost. Rows that lose the whole entry score 0, being below the allowance.
- **The cost on those rows.** Their stored results no longer sum to zero: their off-argmax entries flush when written.
    - A Σg metric without an allowance for that flushed mass (28/10/2026) went from 0 to 84.8 ε (fp32) and 5.65·10⁹ ε (fp64).
    - The harness's Σg metric now allows for it (the mass of the exact entries below the normal range), and reads 0.
- **Why two branches.** A single-branch form, with factors of exactly 1 off the scaled rows, was as accurate on the CPU. On the GPU, though, XLA rounded other rows' VJPs differently: 4 keys worse against the two-branch form, none better. For example, `margin_bigk` fp64 softmax VJP worst 4.72 against 2.71 ε.

### Where the defaults are less accurate than the paper's code

**The zero-sum residual Σg is the trade-off.** The exact VJPs sum to zero, but the correctly rounded gradient does not: its Σg reaches 35 ε on fp64 `flat` (2 048 classes) and 67 ε on bf16 `margin_bigk` (variants harness).

- The paper's rule makes one entry minus the sum of the others, so its results sum to zero by construction; that entry then cancels on flat rows (6.9·10⁴ ε at 32 000 classes; 37–50 ε on the suites' 2 048-class `flat` rows).
- The defaults keep every entry accurate instead. Their sums are therefore those of accurate entries, which on some keys sit further from zero than the paper's code's.

The results are not projected onto Σ = 0. Subtracting the row mean shifts every entry by about ε·Σ|g|/K, which destroys entries much smaller than that (see "Further figures quoted in the docstrings").

**Forward mode under flush-to-zero arithmetic, `rule="accurate"`.** On JAX CPU, the flush-band rows above. In torch, whose default is `"accurate"`, only under `torch.set_flush_denormal(True)`: there the accurate rule loses 4 551 ε on the suite's fp32 flush-band rows and `rule="flush_proof"` 0; the paper's port was not measured there, so its being ahead in torch is inferred from JAX.

**`rule="paper"`** gives the paper's zero-sum structure on this library's kernels. Against the paper's code over 20 seeds:

- **JAX** (the harness arm equal to the library's paper rule except for the cross-entropy gradient, which is the library's target pivot under every rule): worse only on Σ keys.
    - GPU: nine bfloat16 VJP-sum keys (for example, `flat` log_softmax Σg mean 15.5 against 0.04 ε) and one float16 key, mixed.
    - CPU: `flat` bfloat16 log_softmax Σg (mean 15.5 against 11.8 ε), and one fp64 key flagged on its mean only (`margin_bigk` log_softmax Σg, +0.17 ε, equal worst).
    - Mechanism: 16-bit inputs are computed in float32, and the pivot is taken there. Each entry is then cast to 16 bits separately, so the stored row sums to the casting errors. The paper's code reaches near-zero on JAX GPU apparently because XLA keeps float32 inside its bfloat16 fusions; this was inferred, not checked in the compiled program.
- **torch** (server, 29/10/2026): better on 202 / 205 keys, worse on 6 / 2, mixed on 5 / 2 (CPU / CUDA).
    - On the VJP sums, its mean is worse than the paper's code's on no key. The default's mean is worse on 15 keys on CPU, all VJP sums, and on 18 on CUDA: 16 VJP sums and two componentwise JVP keys (`margin_bigk` fp32 log_softmax and softmax JVPs, means 515 against 309 and 650 against 446 ulp).
    - Every "worse" key is a worst-case flag with no worse mean. On CPU, five are `safe` componentwise JVP/VJP keys where both sides sit at thousands of ulp (up to 4.47·10⁵ against 5.37·10⁴), and the sixth is `flat` fp32 softmax JVP Σ (1.52·10³ against 933 ε). On CUDA both are `safe` componentwise keys (log_softmax VJP fp32 7.75·10⁴ against 3.35·10⁴; softmax VJP fp64 4.47·10⁵ against 5.37·10⁴).
    - Every "mixed" key has the better mean, but two have a worse worst case on a metric the suites check: CPU `wide` fp32 softmax VJP 704 against 320 ε, beyond the suites' clause (1.25 × 320 + 4 = 404), and `safe` fp64 13.7 against 8.54 ε (CUDA 11.9 against 7.47 ε). The suite passes `wide` fp32 because its single seed does not reach that row.

### Other known limits

- **Probability-label cross-entropy.** The gradient Σp·q − p subtracts two rounded probabilities: on one fp32 `margin` row in each framework (different seeds of 201–203, CPU, the library of 27/10/2026), 12.6 ε (JAX) and 4.92 ε (torch). The previous rules gave 4.55 and 2.89 ε there; the previous and current libraries' q_max differ by under 1 ulp. Teacher logits (`soft_target_cross_entropy`) avoid it.
- **JAX bf16 `outlier`, KL teacher gradient: 124 ε.** The gradient comes from a teacher probability of 9.2·10⁻³⁹, which XLA CPU flushes. Torch, which does not flush, gets 0.46 ε; the paper's code and native give 10⁴⁰ ε and inf.

## Further figures quoted in the docstrings

| Docstring                              | Figure and source                                                                                                                                                                                                                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Flush band (`softmaxfix.jax`)          | Logit gaps near 79 (fp32) and 700 (fp64) at 4 096 classes: −ln(smallest normal) − ln K.                                                                                                                                                                                                  |
| Projection, JAX                        | Projecting the log_softmax VJP projects the forward-mode input too: log_softmax JVP mean +0.05–0.09 ε on 11–12 keys, worst 0.455 → 0.907 ε on fp32 `flat` (candidates harness, 27/10/2026).                                                                                              |
| Projection, torch                      | Torch CPU, 5 seeds (laptop): the gradient of q_max on fp64 `wide` 2·10¹⁵⁸ ulp projected, 4.4 unprojected; the log_softmax VJP componentwise on fp32 `margin_bigk` 1.8·10⁵ ulp projected, 303 unprojected. Projection's gain: Σg worst 7.2 → 3.0 ε on fp64 `safe` (torch CUDA, the current defaults; candidates harness). |
| Gradient of q_max (`softmaxfix.torch`) | The softmax VJP at the one-hot cotangent of the argmax, every regime (5 seeds CPU, 3 CUDA; laptop): within 6.74 ulp (CPU) and 5.53 ulp (CUDA); the paper's code reaches 510 ulp and native 9.0·10¹⁵.                                                                                     |
| `flush_proof` in torch                 | Under `torch.set_flush_denormal(True)`, on the suite's fp32 flush-band rows: `rule="accurate"` 4 551 ε, `rule="flush_proof"` 0.                                                                                                                                                          |
| Single-branch cost (`softmaxfix.jax`)  | ×0.40–0.92 of the two-branch form's time: the harness's single-branch arm against its two-branch arm, in the runs of the "Cost" table (median of per-repetition ratios).                                                                                                                                                                      |

## Probes cited in the code

| Code                                                         | Probe and result                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_log_mass`; JAX integer-label CE with the target masked out | 600 rows per format at logit spreads 10, 100 and 300, against mpmath. `_log_mass`, as the difference of two logsumexps: worst 97 ulp → 2.0–3.4 ulp. The masked-target CE value, in optax's form: 57 / 171 → 10.6 / 9 ulp (fp32 / fp64).                                                                                               |
| The same CE's derivatives                                    | Autodiff of the exact-shift value m + log1p(s) gives the argmax entry as 1 − s/(1 + s), which cancels on flat rows (logits N(0, 0.1²)). Worst: 883 / 1.06·10³ ε at K = 1 024 and 2.7·10⁴ / 2.5·10⁴ ε at K = 32 768 (fp32 / fp64), against 0.78–1.19 ε for optax's form. The library therefore keeps the logsumexp form's derivatives. |
| Centred softmax cotangent                                    | Numpy, fp32 `wide`: worst 410 ε with r·v_max at the argmax, 151 ε uncentred with the argmax entry as −Σ of the others, 1.8 ε centred. On flat rows centring is harmful (Σg 467 ε), hence confident rows only.                                                                                                                         |
| Softmax argmax entry under XLA CPU flushing                  | Fp32 `margin_bigk`: the off-argmax entries q_j·v_j − q_j·⟨q, v⟩ flush subnormal products that all carry the same sign. Summing 4 095 of them into the pivot gives 5.4·10⁴ ε, a 0.9 % relative error on a normal-range value (eager evaluation; under `jit` the loss depends on how XLA compiles the sum, as in the table above). Neither torch nor JAX GPU shows it.                                                                      |

## Cost

All costs are forward + backward times as ratios, from interleaved repetitions (the median of per-repetition ratios). Absolute CUDA times on the laptop varied ×9 between runs; only the interleaved ratios are reliable.

**Against native** (laptop; the library of 27/10/2026, when JAX's default was `rule="accurate"`; fp32 unless stated):

| Implementation                  | log_softmax | softmax | cross-entropy | soft-target CE | KL    |
| ------------------------------- | ----------- | ------- | ------------- | -------------- | ----- |
| torch CUDA, 1 024 × 32 768      | ×9.9        | ×3.6    | ×11.5         | ×16.9          | ×19.7 |
| torch CUDA, bf16                | ×26.4       | ×11.5   | ×30.2         | ×42.9          | ×46.3 |
| torch CPU, 128 × 32 768         | ×10.5       | ×4.0    | ×12.2         | ×19.5          | ×13.5 |
| JAX GPU (`jit`), 1 024 × 32 768 | ×1.5        | ×1.6    | ×2.0          | ×5.4           | ×7.7  |
| JAX GPU (`jit`), bf16           | ×3.0        | ×3.2    | ×3.3          | ×6.3           | ×8.9  |

This library computes 16-bit inputs in float32; native torch on CUDA keeps bf16 temporaries.

**Options against the defaults of 27/10/2026** (fp32; laptop: RTX 3050 Ti at 2 048 × 32 768, CPU at 256 × 32 768). The JAX default was then `rule="accurate"`. Each figure is the option's time divided by the defaults':

|                     | torch CUDA | torch CPU  | JAX GPU    | JAX CPU    |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| previous rules      | ×0.47–0.52 | ×0.47–0.59 | ×0.77–0.87 | ×0.98–1.05 |
| `exact_shift=False` | ×0.49–0.65 | ×0.55–0.71 | ×0.76–0.88 | ×0.97–1.01 |
| `rule="paper"`      | ×0.78–0.97 | ×0.72–0.97 | ×0.69–0.89 | ×0.47–0.90 |
| native              | ×0.09–0.25 | ×0.09–0.18 | ×0.46–0.75 | ×0.25–0.38 |

- In torch eager, the defaults cost ×1.7–2.1 the previous rules' time. In log_softmax and cross-entropy nearly all of it is the exact shift's extra elementwise kernels; in softmax the exact shift accounts for ×1.4–1.5 of ×2.1, and the centred backward for the rest. XLA fuses the extra kernels under `jit`.
- Peak CUDA memory in torch is unchanged by the defaults for log_softmax and CE (1 600 MiB). The softmax backward goes from 1 280 to 1 536 MiB.

**JAX `rule="flush_proof"`** (server: one GTX 1080 Ti, and 8 pinned CPU cores, the machine otherwise idle; per-repetition random order, with a null arm timing the library twice).

- The rows are softmax forward + backward medians, in ms.
- "Accurate" and "two-branch" are the harness's implementations of the two rules. "Library" is this library, timed alongside, then still in its single-branch form.
- The ratio column is the median of per-repetition ratios, not the ratio of the medians.

| Cell                      | accurate (ms) | two-branch (ms) | library, single-branch (ms) | two-branch / accurate |
| ------------------------- | ------------- | --------------- | --------------------------- | --------------------- |
| CPU 256 × 32768 float32   | 81.39         | 103.84          | 82.31                       | ×1.32                 |
| CPU 2048 × 1024 float32   | 7.63          | 19.45           | 7.92                        | ×2.56                 |
| GPU 2048 × 32768 bfloat16 | 5.29          | 9.74            | 8.27                        | ×1.84                 |
| GPU 2048 × 32768 float32  | 9.56          | 12.60           | 11.08                       | ×1.32                 |
| GPU 16384 × 1024 float32  | 2.51          | 3.78            | 3.49                        | ×1.51                 |

- log_softmax and cross-entropy run as under the accurate rule, within 1 %.
- The two-branch form now shipped equals the harness's two-branch arm bitwise on the CPU (seed 1; `safe`, `flush_bigk` and `cliff`). The shipped library itself has not been timed in that form, on CPU or GPU.

## The paper's experiments, rerun

**Grokking** (on the server; {JAX, torch} × {bf16, fp32, fp64} × ε ∈ {10⁻⁸, 10⁻¹², 10⁻¹⁶}, 5 seeds × 30 000 epochs).

- The task (a + b mod 31, 50 % training split) and the optimiser (full-batch AdamW, η = 10⁻³, β = (0.9, 0.98), λ = 1) follow the paper.
- The paper specifies only the model family. The model details are this project's choice: a 1-layer width-64 ReLU Transformer, 4 heads, MLP 256, learned positions.
- Margins are read at 30 000 epochs; the paper's Table 1 reads them at 10 000.
- The "this library" rows are the library of 27/10/2026, when JAX's default was `rule="accurate"`.

Final margins, fp32, medians over 5 seeds:

| ε                   | 10⁻⁸  | 10⁻¹² | 10⁻¹⁶ |
| ------------------- | ----- | ----- | ----- |
| Paper (Table 1)     | 17.9  | 27.4  | 36.8  |
| Paper's code, JAX   | 17.16 | 26.73 | 36.08 |
| Paper's code, torch | 17.17 | 26.19 | 35.68 |
| This library, JAX   | 17.15 | 26.63 | 36.21 |
| This library, torch | 17.17 | 26.21 | 35.66 |

- In fp32 and fp64, neither this library nor the paper's code ever reaches an exact-zero loss or spikes.
- Native reaches an exact zero in every fp32 run, with up to 43 (JAX) and 35 (torch) loss spikes.
- In bf16 no run of any loss groks (test accuracy 37–48 %), where the paper reports 93–100 %. Here the whole run is in bf16, parameters and optimiser state included. The paper does not describe its bf16 setup in that detail, so this is a setup difference, not a verdict on any loss.

**Distillation.** At z_S = z_T this library gives a gradient of exactly 0, a held-out KL of exactly 0 and no drift, in every framework, format and JAX precision mode. The paper's code does the same, except in JAX bf16 under XLA's excess precision.

Native's gradient there is not zero: 9.3·10⁻⁹ / 9.6·10⁻⁹ in fp32 (JAX / torch). In fp32 its KL floor is 1.05·10⁻⁵–1.31·10⁻⁵ and its drift 3.1–3.2 %; across formats, in the default precision modes, 1.05·10⁻⁵–6.8·10⁻⁵ and 3–6 %. Under exact bf16 semantics in JAX (no excess precision), native's KL floor is 1.0·10⁻² and its drift 80 %.
