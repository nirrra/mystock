from __future__ import annotations

import pytest

from stocks_analyzer.lgbm_utils import build_lgbm_params, fit_lgbm_with_device


def test_build_lgbm_params_adds_gpu_device_and_jobs() -> None:
    params = build_lgbm_params(
        {"learning_rate": 0.1, "n_jobs": 1},
        device="cuda",
        n_jobs=-1,
        gpu_device_id=0,
    )

    assert params["device_type"] == "cuda"
    assert params["gpu_device_id"] == 0
    assert params["n_jobs"] == -1


def test_fit_lgbm_with_auto_device_falls_back_to_cpu() -> None:
    attempts: list[str] = []

    class FakeEstimator:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            device = self.params.get("device_type", "cpu")
            attempts.append(device)
            if device != "cpu":
                raise RuntimeError(f"{device} unavailable")
            return self

    model, used_device = fit_lgbm_with_device(FakeEstimator, {}, [[1]], [1], device="auto", n_jobs=2)

    assert used_device == "cpu"
    assert model.params["n_jobs"] == 2
    assert attempts == ["cuda", "cpu"]


def test_fit_lgbm_with_bad_device_rejects_value() -> None:
    with pytest.raises(ValueError):
        build_lgbm_params({}, device="tpu")


def test_fit_lgbm_with_explicit_cuda_can_fallback_to_cpu() -> None:
    attempts: list[str] = []

    class FakeEstimator:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            device = self.params.get("device_type", "cpu")
            attempts.append(device)
            if device == "cuda":
                raise RuntimeError("cuda unavailable")
            return self

    _, used_device = fit_lgbm_with_device(FakeEstimator, {}, [[1]], [1], device="cuda", n_jobs=-1)

    assert used_device == "cpu"
    assert attempts == ["cuda", "cpu"]
