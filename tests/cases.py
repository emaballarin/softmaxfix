"""Deterministic float64 test inputs shared by the JAX and PyTorch suites (rounded to the format under test later)."""

import zlib

import numpy as np

# Margins up to just past the point where e^{-m} underflows each format, so sweeps cross every precision cliff.
MAX_MARGIN = {"float16": 17.5, "bfloat16": 93.0, "float32": 104.0, "float64": 745.0}

# Student–teacher gaps of the near-teacher regime.
GAPS = (1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-12)

# Hand-made soft-target rows (student logits, teacher logits) probing supports, masks and underflow.
EDGE_PAIRS = {
    "teacher gap 120, student gap 90": ([0.0, -30.0], [0.0, -120.0]),
    "teacher gap 120, equal student": ([0.0, 0.0], [0.0, -120.0]),
    "teacher subnormal probability": ([0.0, 0.0], [0.0, -95.0]),
    "teacher masked": ([0.0, 1.0, -5.0], [0.0, 1.0, -np.inf]),
    "both masked": ([0.0, 1.0, -np.inf], [0.0, 1.0, -np.inf]),
    "student masked": ([0.0, 1.0, -np.inf], [0.0, 1.0, -5.0]),
    "one class left": ([3.0, -np.inf, -np.inf], [-1.0, -np.inf, -np.inf]),
}


# A student probability that underflows in each format on a class the teacher supports: the KL stays finite.
UNDERFLOW_ROWS = {
    "float16": ([0.0, -20.0], [0.0, -1.0]),
    "bfloat16": ([0.0, -95.0], [0.0, -1.0]),
    "float32": ([0.0, -110.0], [0.0, -1.0]),
    "float64": ([0.0, -800.0], [0.0, -1.0]),
}


def rng(seed: int, *tags: str) -> np.random.Generator:
    """Generator derived from the master seed and stable string tags (one per stochastic concern)."""
    return np.random.default_rng(np.random.SeedSequence([seed, *(zlib.crc32(t.encode()) for t in tags)]))


def random_logits(gen: np.random.Generator, n: int, k: int, scale: float, offset: float = 0.0) -> np.ndarray:
    """Gaussian logits of the given scale and offset."""
    return gen.normal(size=(n, k)) * scale + offset


def _scatter_target(gen: np.random.Generator, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle each row's columns so the target (column 0) lands at a random index."""
    n, k = z.shape
    perm = np.argsort(gen.random((n, k)), axis=1)
    out = np.take_along_axis(z, perm, axis=1)
    t = np.argmax(perm == 0, axis=1)
    return out, t


def margin_sweep(
    gen: np.random.Generator, fmt: str, n: int = 48, k: int = 4, max_margin: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose target is the argmax with logit margin m from 0 to past underflow of e^{-m} in ``fmt``
    (or to ``max_margin``, if smaller)."""
    margins = np.linspace(0.0, min(MAX_MARGIN[fmt], max_margin or np.inf), n)
    others = gen.normal(size=(n, k - 1))
    lse = np.log(np.exp(others).sum(axis=1))
    return _scatter_target(gen, np.concatenate([(margins + lse)[:, None], others], axis=1))


def flat(gen: np.random.Generator, n: int = 8, k: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Near-uniform logits over many classes (as at initialisation), target never the argmax."""
    z = gen.normal(size=(n, k)) * 0.1
    t = (np.argmax(z, axis=1) + 1 + gen.integers(0, k - 1, n)) % k
    return z, t


def wrong_class(gen: np.random.Generator, n: int = 48, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose target sits up to 60 below the confidently predicted class (large negative margin)."""
    others = gen.normal(size=(n, k - 1))
    others[:, 0] += np.linspace(0.0, 60.0, n)
    target = gen.normal(size=n)
    return _scatter_target(gen, np.concatenate([target[:, None], others], axis=1))


def near_teacher(
    gen: np.random.Generator, gaps: tuple[float, ...], n: int = 16, k: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Student logits z_T + gap·ε for each gap (n rows per gap), teacher logits z_T ~ N(0, 2²)."""
    zt = gen.normal(size=(n * len(gaps), k)) * 2.0
    g = np.repeat(np.asarray(gaps), n)[:, None]
    return zt + g * gen.normal(size=zt.shape), zt


def near_teacher_offset(
    gen: np.random.Generator,
    offsets: tuple[float, ...] = (4.0, 16.0, 64.0),
    gaps: tuple[float, ...] = (1e-1, 1e-3, 1e-6),
    n: int = 8,
    k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Student logits z_T + c + gap·ε: near the teacher up to a common offset c (softmax ignores it)."""
    groups = [(c, g) for c in offsets for g in gaps]
    zt = gen.normal(size=(n * len(groups), k)) * 2.0
    c = np.repeat([c for c, _ in groups], n)[:, None] * gen.choice([-1.0, 1.0], size=(n * len(groups), 1))
    g = np.repeat([g for _, g in groups], n)[:, None]
    return zt + c + g * gen.normal(size=zt.shape), zt


def outlier(gen: np.random.Generator, n: int = 48, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Student near the teacher except on one class the teacher considers very unlikely (40–90 below its other
    logits) but on which the student keeps ordinary mass."""
    zt = gen.normal(size=(n, k)) * 2.0
    zs = zt + 1e-3 * gen.normal(size=zt.shape)
    j = gen.integers(0, k, n)
    rows = np.arange(n)
    zt[rows, j] = zt.min(axis=1) - np.linspace(40.0, 90.0, n)
    zs[rows, j] = gen.normal(size=n)
    return zs, zt
