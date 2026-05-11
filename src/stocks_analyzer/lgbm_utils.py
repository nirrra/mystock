from __future__ import annotations

import logging
from typing import Any, Callable


LGBM_DEVICE_CHOICES = {"cpu", "gpu", "cuda", "auto"}


def normalized_lgbm_device(device: str | None) -> str:
    normalized = str(device or "cpu").strip().lower()
    if normalized not in LGBM_DEVICE_CHOICES:
        raise ValueError(f"Unsupported LightGBM device: {device}. Use cpu, gpu, cuda, or auto.")
    return normalized


def build_lgbm_params(
    base_params: dict[str, Any],
    *,
    device: str = "cpu",
    n_jobs: int | None = None,
    gpu_platform_id: int | None = None,
    gpu_device_id: int | None = None,
) -> dict[str, Any]:
    params = dict(base_params)
    if n_jobs is not None:
        params["n_jobs"] = int(n_jobs)
    normalized = normalized_lgbm_device(device)
    if normalized in {"gpu", "cuda"}:
        params["device_type"] = normalized
        if gpu_platform_id is not None:
            params["gpu_platform_id"] = int(gpu_platform_id)
        if gpu_device_id is not None:
            params["gpu_device_id"] = int(gpu_device_id)
    else:
        params.pop("device_type", None)
        params.pop("device", None)
        params.pop("gpu_platform_id", None)
        params.pop("gpu_device_id", None)
    return params


def fit_lgbm_with_device(
    estimator_factory: Callable[..., Any],
    base_params: dict[str, Any],
    X: Any,
    y: Any,
    *,
    device: str,
    n_jobs: int | None = None,
    gpu_platform_id: int | None = None,
    gpu_device_id: int | None = None,
    fallback_to_cpu: bool = True,
    fit_label: str = "LightGBM",
) -> tuple[Any, str]:
    normalized = normalized_lgbm_device(device)
    # On Windows, LightGBM's OpenCL `gpu` backend may fail before Python can
    # recover if Boost.Compute cannot create its cache directory. `auto` is
    # intentionally conservative: try CUDA first, then a safe CPU fallback.
    if normalized == "auto":
        candidates = ("cuda", "cpu")
    elif normalized in {"cuda", "gpu"} and fallback_to_cpu:
        candidates = (normalized, "cpu")
    else:
        candidates = (normalized,)
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            params = build_lgbm_params(
                base_params,
                device=candidate,
                n_jobs=n_jobs,
                gpu_platform_id=gpu_platform_id,
                gpu_device_id=gpu_device_id,
            )
            model = estimator_factory(**params)
            return model.fit(X, y), candidate
        except Exception as exc:
            last_exc = exc
            if candidate == "cpu":
                break
            if not fallback_to_cpu and normalized != "auto":
                raise
            logging.warning(
                "%s failed with LightGBM device_type=%s; trying next device. Error: %s",
                fit_label,
                candidate,
                exc,
            )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{fit_label} LightGBM fit failed without an exception.")
