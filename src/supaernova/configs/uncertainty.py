# Copyright 2025 Patrick Armstrong
"""One description of the measurement noise and model error, shared by every step.

The data, sim and posterior steps each carry an `uncertainty` table, resolved
into an `UncertaintyModel`. It says how reported uncertainties are calibrated,
how bands are integrated, how the PAE's reconstruction (model) error is measured
and scaled, and how both are correlated in the likelihood. A step that consumes
another step's results checks that the two agree, so data, sims and posteriors
cannot silently assume different things.

Presets reproduce earlier implementations numerically:

* ``main``: the pre-rewrite code (main, d891470). σ = 1.4 σ_raw + 4e-10 before
  bands are integrated, model error r·√(m² + σ²), every datum independent except
  that model error is coherent within a band.
* ``dev``: the uncertainty rewrite (dev, 94e5e83). σ = σ_raw, model error r·|m|,
  one Gaussian per epoch with a rank-1 model-error correlation ρ.
* ``proposal``: a per-SN Gaussian whose noise, floor and model error are
  correlated in wavelength and time, with every parameter estimated from the
  PAE's held-out residuals (``"auto"``).
* ``proposal_fixed``: ``proposal`` with the parameters fixed to the values fitted
  in the uncertainty study (base PAE, held-out SNe).

Keys given alongside ``preset`` override the preset (deep merge), e.g.
``uncertainty = { preset = "dev", correlation = { rho = 0.4 } }``.

The legacy flat options (`sigma_scale`, `sigma_floor`, `recon_correlation`,
`fractional_error`, `weighted_error`, `measurement_error`,
`reconstruction_error`, `inject_recon_error`) are still accepted. They imply the
``dev`` preset and cannot be combined with an `uncertainty` table.
"""

from copy import deepcopy
from typing import Any, Literal

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from supaernova.configs.base import BaseConfig

Auto = Literal["auto"]
Preset = Literal["main", "dev", "proposal", "proposal_fixed"]
DEFAULT_PRESET: Preset = "dev"


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value == "auto"


class Calibration(BaseConfig):
    """Calibration of reported uncertainties: σ = scale·σ_raw (+) floor."""

    scale: PositiveFloat = 1.0
    floor: NonNegativeFloat = 0.0
    # "linear": σ = scale·σ_raw + floor; "quadrature": σ = √((scale·σ_raw)² + floor²)
    floor_mode: Literal["linear", "quadrature"] = "linear"
    # "after_bands": calibrate every datum once rows are built (bands from raw σ);
    # "before_bands": calibrate pixels first, so bands inherit calibrated σ (main)
    stage: Literal["after_bands", "before_bands"] = "after_bands"


class Bands(BaseConfig):
    """How synthetic bands are integrated and which are kept."""

    # A synthetic band exists only if every pixel its filter covers is valid.
    # "all_pixels": laser-line pixels count as invalid (dev); "ignore_laser": they
    # don't, and their flux is integrated (main, which only masked laser lines in
    # spectra)
    validity: Literal["all_pixels", "ignore_laser"] = "all_pixels"
    # Drop, in the posterior, bands whose filter reaches a pixel outside the
    # fitted wavelength range (main's `biased_phot`)
    drop_partial: bool = False
    # Band σ from pixel σ: "independent" √(Σ g² σ²), or "kernel" with the noise
    # kernel's pixel-to-pixel correlation
    noise_propagation: Literal["independent", "kernel"] = "independent"


class ModelError(BaseConfig):
    """How the PAE's reconstruction error r(λ, t) is measured and scaled."""

    # Model error e = r·scale: "model" |m|, "model_and_noise" √(m² + σ²) (main),
    # or "constant" 1
    scale: Literal["model", "model_and_noise", "constant"] = "model"
    # r² estimator: "ratio_of_sums" Σw²(res² − σ²)/Σw²·scale² (dev), "mean_ratio"
    # Σw²(res² − σ²)/scale² / Σw² (main), or "kernel_ml" (joint fit with the kernel)
    estimator: Literal["ratio_of_sums", "mean_ratio", "kernel_ml"] = "ratio_of_sums"
    weighted: bool = True  # weights w = 1/σ
    subtract_noise: bool = True  # subtract σ² from the squared residuals
    # Which SNe' residuals measure r ("match" = the subset being fitted)
    subset: Literal["train", "test", "combined", "match"] = "combined"
    time_bin: PositiveFloat = 0.1  # in normalised time


class Correlation(BaseConfig):
    """How data are grouped into Gaussians and how the model error correlates."""

    # "independent": every datum on its own (main); "epoch": one Gaussian per
    # epoch with rank-1 model-error correlation `rho` (dev); "kernel": one
    # Gaussian per SN with the wavelength / time kernels (proposal)
    mode: Literal["independent", "epoch", "kernel"] = "epoch"
    # Independent mode: model error within a band either averages down
    # ("independent_pixels") or is fully coherent ("coherent", main)
    band_coherence: Literal["independent_pixels", "coherent"] = "independent_pixels"
    rho: float = Field(default=0.0, ge=0.0, le=1.0)  # epoch mode


class KernelWavelength(BaseConfig):
    """k(Δλ) = w0 δ + w_s Matérn-3/2(Δλ; ℓ_s) + w_l L(Δλ; ℓ_l), Δλ in pixels."""

    w0: float | Auto = "auto"
    w_s: float | Auto = "auto"
    ell_s: float | Auto = "auto"
    w_l: float | Auto = "auto"
    ell_l: float | Auto = "auto"
    long_form: Literal["se", "exp"] = "se"

    @model_validator(mode="after")
    def _validate(self):
        for name in ("w0", "w_s", "w_l"):
            value = getattr(self, name)
            if isinstance(value, (int, float)) and not 0 <= value <= 1:
                self._raise(f"`{name}` = {value} must be in [0, 1]")
        for name in ("ell_s", "ell_l"):
            value = getattr(self, name)
            if isinstance(value, (int, float)) and value <= 0:
                self._raise(f"`{name}` = {value} must be positive")
        weights = [
            w for w in (self.w0, self.w_s, self.w_l) if isinstance(w, (int, float))
        ]
        if len(weights) == 3 and abs(sum(weights) - 1) > 1e-3:
            self._raise(f"w0 + w_s + w_l = {sum(weights)} must be 1")
        return self


class KernelTime(BaseConfig):
    """k(Δt) = (1 − w_t) δ + w_t exp(−|Δt| / τ), Δt in days between epochs."""

    w_t: float | Auto = "auto"
    tau: float | Auto = "auto"

    @model_validator(mode="after")
    def _validate(self):
        if isinstance(self.w_t, (int, float)) and not 0 < self.w_t <= 1:
            self._raise(f"`w_t` = {self.w_t} must be in (0, 1]")
        if isinstance(self.tau, (int, float)) and self.tau <= 0:
            self._raise(f"`tau` = {self.tau} must be positive")
        return self


class KernelTerm(BaseConfig):
    """One variance term: its amplitude and its wavelength and time kernels."""

    # Noise: the scale a on σ; floor: f; model: unused (r(λ, t) sets it)
    amplitude: float | Auto = "auto"
    n_modes: NonNegativeInt | None = None  # overrides `kernel.n_modes`; 0 = none
    on_bands: bool = True  # whether the term also applies to bands
    wavelength: KernelWavelength = Field(default_factory=KernelWavelength)
    time: KernelTime | None = Field(default_factory=KernelTime)  # None = white


class Kernel(BaseConfig):
    """Kernel likelihood settings (correlation.mode = "kernel")."""

    bin_width: PositiveInt = 3  # spectra are averaged into super-pixels of this width
    # Long-range wavelength modes per term. 5 (and none for the noise, whose
    # long-range part is weak) keep σ(ΔM) within ~2% of 10 modes on the base
    # sample at ~1/16 of the cost.
    n_modes: PositiveInt = 5
    # Leading eigenvectors kept of each SN's epoch-time kernel (None = all). The
    # low-rank inner system has (time modes × wavelength modes) columns, so this
    # bounds its size independently of the number of epochs; each epoch's dropped
    # time variance is restored on the block diagonal.
    time_modes: PositiveInt | None = 6
    # Correlate the local (short-range) parts between epochs too. False keeps them
    # independent between epochs, so B is diagonal: far cheaper, slightly less
    # exact (long-range parts keep their time correlation either way)
    local_time: bool = True
    jitter: NonNegativeFloat = 1e-6
    # float32 + XLA is ~6x faster than float64 on consumer GPUs; log-likelihoods
    # agree to ~1e-4 relative (tests/test_kernel_likelihood.py)
    dtype: Literal["float32", "float64"] = "float32"
    jit: bool = True  # XLA-compile the kernel likelihood
    sn_batch: PositiveInt | None = None
    noise: KernelTerm = Field(
        default_factory=lambda: KernelTerm(time=None, on_bands=True, n_modes=0)
    )
    floor: KernelTerm = Field(default_factory=KernelTerm)
    model: KernelTerm = Field(default_factory=KernelTerm)

    @model_validator(mode="after")
    def _noise_is_white_in_time(self):
        # Measurement noise is independent between epochs by definition
        self.noise.time = None
        return self


class SimUncertainty(BaseConfig):
    """How the sim injects what the posterior assumes."""

    inject: bool = True  # inject model error
    # Fit the sim's noise model to "raw" reported σ (dev), or to "calibrated" σ (main)
    noise_fit: Literal["raw", "calibrated"] = "raw"
    # Bands from the noise-free model with their own noise ("independent"), or
    # integrated from the noisy spectrum ("from_noisy_spectrum", main)
    bands: Literal["independent", "from_noisy_spectrum"] = "independent"
    draw: Literal["kronecker", "dense"] = "kronecker"  # kernel-mode draws


_PROPOSAL_FIXED_KERNEL = {
    "noise": {
        "amplitude": 1.33,
        "time": None,
        "n_modes": 0,
        "wavelength": {
            "w0": 0.46,
            "w_s": 0.41,
            "ell_s": 0.7,
            "w_l": 0.13,
            "ell_l": 53.0,
        },
    },
    "floor": {
        "amplitude": 4.3e-3,
        "wavelength": {
            "w0": 0.18,
            "w_s": 0.21,
            "ell_s": 4.2,
            "w_l": 0.61,
            "ell_l": 115.0,
        },
        "time": {"w_t": 1.0, "tau": 8.6},
    },
    "model": {
        "amplitude": 1.0,
        "wavelength": {
            "w0": 0.0,
            "w_s": 0.60,
            "ell_s": 4.2,
            "w_l": 0.40,
            "ell_l": 100.0,
        },
        "time": {"w_t": 0.86, "tau": 11.6},
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    "main": {
        "calibration": {
            "scale": 1.4,
            "floor": 4e-10,
            "floor_mode": "linear",
            "stage": "before_bands",
        },
        "bands": {
            "validity": "ignore_laser",
            "drop_partial": True,
            "noise_propagation": "independent",
        },
        "model_error": {
            "scale": "model_and_noise",
            "estimator": "mean_ratio",
            "weighted": True,
            "subtract_noise": True,
            "subset": "combined",
        },
        "correlation": {"mode": "independent", "band_coherence": "coherent"},
        "sim": {
            "inject": False,
            "noise_fit": "calibrated",
            "bands": "from_noisy_spectrum",
        },
    },
    "dev": {
        "calibration": {
            "scale": 1.0,
            "floor": 0.0,
            "floor_mode": "linear",
            "stage": "after_bands",
        },
        "bands": {
            "validity": "all_pixels",
            "drop_partial": False,
            "noise_propagation": "independent",
        },
        "model_error": {
            "scale": "model",
            "estimator": "ratio_of_sums",
            "weighted": True,
            "subtract_noise": True,
            "subset": "combined",
        },
        "correlation": {"mode": "epoch", "rho": 0.0},
        "sim": {"inject": True, "noise_fit": "raw", "bands": "independent"},
    },
    "proposal": {
        "calibration": {
            "scale": 1.0,
            "floor": 0.0,
            "floor_mode": "linear",
            "stage": "after_bands",
        },
        "bands": {
            "validity": "all_pixels",
            "drop_partial": False,
            "noise_propagation": "kernel",
        },
        "model_error": {
            "scale": "model",
            "estimator": "kernel_ml",
            "weighted": True,
            "subtract_noise": True,
            "subset": "test",
        },
        "correlation": {"mode": "kernel"},
        "sim": {
            "inject": True,
            "noise_fit": "raw",
            "bands": "independent",
            "draw": "kronecker",
        },
    },
}
PRESETS["proposal_fixed"] = deepcopy(PRESETS["proposal"])
PRESETS["proposal_fixed"]["kernel"] = deepcopy(_PROPOSAL_FIXED_KERNEL)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


class UncertaintyModel(BaseConfig):
    preset: Preset = DEFAULT_PRESET
    calibration: Calibration = Field(default_factory=Calibration)
    bands: Bands = Field(default_factory=Bands)
    model_error: ModelError = Field(default_factory=ModelError)
    correlation: Correlation = Field(default_factory=Correlation)
    kernel: Kernel = Field(default_factory=Kernel)
    sim: SimUncertainty = Field(default_factory=SimUncertainty)

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        if data is None:
            data = {}
        if isinstance(data, dict):
            data = {k.lower(): v for k, v in data.items()}
            preset = data.get("preset", DEFAULT_PRESET)
            if preset not in PRESETS:
                cls._raise(
                    f"Unknown uncertainty preset {preset!r}; use one of {list(PRESETS)}"
                )
            data = _deep_merge(PRESETS[preset], data)
            data["preset"] = preset
        return data

    @model_validator(mode="after")
    def _validate(self):
        mode = self.correlation.mode
        if mode != "epoch" and self.correlation.rho != 0:
            self._raise("`correlation.rho` only applies to correlation.mode = 'epoch'")
        if (
            mode != "independent"
            and self.correlation.band_coherence != "independent_pixels"
        ):
            self._raise(
                "`correlation.band_coherence` only applies to correlation.mode = 'independent'"
            )
        if mode == "kernel":
            if (self.calibration.scale, self.calibration.floor) != (1.0, 0.0):
                self._raise(
                    "With correlation.mode = 'kernel', the noise scale and floor are "
                    "`kernel.noise.amplitude` and `kernel.floor.amplitude`; keep "
                    "`calibration` at scale 1, floor 0 so they are not applied twice."
                )
            if self.model_error.estimator != "kernel_ml" and self.has_auto():
                self._raise(
                    "Kernel parameters set to 'auto' need model_error.estimator = 'kernel_ml'"
                )
        elif self.model_error.estimator == "kernel_ml":
            self._raise(
                "model_error.estimator = 'kernel_ml' needs correlation.mode = 'kernel'"
            )
        if self.bands.noise_propagation == "kernel" and mode != "kernel":
            self._raise(
                "bands.noise_propagation = 'kernel' needs correlation.mode = 'kernel'"
            )
        if (
            self.model_error.estimator == "mean_ratio"
            and self.model_error.scale != "model_and_noise"
        ):
            self._raise(
                "model_error.estimator = 'mean_ratio' needs model_error.scale = 'model_and_noise'"
            )
        if (
            self.sim.bands == "from_noisy_spectrum"
            and self.calibration.stage != "before_bands"
        ):
            self._raise(
                "sim.bands = 'from_noisy_spectrum' needs calibration.stage = 'before_bands'"
            )
        return self

    def has_auto(self) -> bool:
        """Whether any kernel parameter is left to be estimated."""
        for term in (self.kernel.noise, self.kernel.floor, self.kernel.model):
            values = [
                term.amplitude,
                *term.wavelength.model_dump(exclude={"name", "long_form"}).values(),
            ]
            if term.time is not None:
                values += list(term.time.model_dump(exclude={"name"}).values())
            if any(_is_auto(v) for v in values):
                return True
        return False

    def signature(self, *sections: str) -> dict[str, Any]:
        """The named sections as plain data, for comparing two steps' models."""
        dump = self.model_dump(exclude={"name"})
        return {s: _strip_names(dump[s]) for s in sections}


def _strip_names(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_names(v) for k, v in value.items() if k != "name"}
    return value


# Legacy flat options and where they live in the uncertainty model
LEGACY_FIELDS: dict[str, tuple[str, str]] = {
    "sigma_scale": ("calibration", "scale"),
    "sigma_floor": ("calibration", "floor"),
    "recon_correlation": ("correlation", "rho"),
    "fractional_error": ("model_error", "scale"),
    "weighted_error": ("model_error", "weighted"),
    "measurement_error": ("model_error", "subtract_noise"),
    "reconstruction_error": ("model_error", "subset"),
    "inject_recon_error": ("sim", "inject"),
}


def pop_legacy_fields(data: dict[str, Any], raise_fn) -> dict[str, Any]:
    """Move legacy flat options into an `uncertainty` table (implying preset dev)."""
    data = dict(data)
    legacy = {k: data.pop(k) for k in list(data) if k in LEGACY_FIELDS}
    if not legacy:
        return data
    if data.get("uncertainty") is not None:
        raise_fn(
            f"Legacy options {sorted(legacy)} cannot be combined with an `uncertainty` "
            "table; move them into it."
        )
    table: dict[str, Any] = {"preset": "dev"}
    for key, value in legacy.items():
        section, field = LEGACY_FIELDS[key]
        if key == "fractional_error":
            value = "model" if value else "constant"
        table.setdefault(section, {})[field] = value
    data["uncertainty"] = table
    return data


def built_with(produced: dict[str, Any] | None) -> "UncertaintyModel | None":
    """The uncertainty model saved in a data or sim result (None for older results)."""
    if produced is None:
        return None
    return UncertaintyModel.model_validate(produced)


def check_consistent(
    consumer: UncertaintyModel,
    produced: dict[str, Any] | None,
    *,
    what: str,
) -> None:
    """Raise if a step would assume a different data construction than it was given.

    Only how the data were built is enforced (calibration, band validity and band
    noise propagation); the model-error and correlation settings may differ on
    purpose, e.g. to fit a sim with a misspecified model.
    """
    producer = built_with(produced)
    if producer is None:
        return
    mine = consumer.signature("calibration", "bands")
    theirs = producer.signature("calibration", "bands")
    for side in (mine, theirs):
        side["bands"].pop("drop_partial", None)  # applied by the posterior only
    if mine != theirs:
        msg = (
            f"The {what} was built with uncertainty {theirs}, but this step assumes "
            f"{mine}. Use the same `uncertainty.calibration` and `uncertainty.bands` "
            "(e.g. the same preset) in both."
        )
        raise ValueError(msg)
