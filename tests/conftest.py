"""Shared pytest configuration: master seed for generated inputs, and GPU memory settings."""

import os

# JAX must not pre-allocate most of the (4 GB) GPU: the PyTorch suite shares it within one session.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest

DEFAULT_SEED = 20261026


def pytest_addoption(parser):
    parser.addoption("--seed", type=int, default=DEFAULT_SEED, help="master seed for generated test inputs")


def pytest_report_header(config):
    return f"softmaxfix master seed: {config.getoption('--seed')}"


@pytest.fixture(scope="session")
def master_seed(request) -> int:
    return request.config.getoption("--seed")
