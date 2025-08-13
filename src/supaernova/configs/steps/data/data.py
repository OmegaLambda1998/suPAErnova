# Copyright 2025 Patrick Armstrong
from typing import Any, Self, Literal, ClassVar, Annotated
from pathlib import Path

import numpy as np
from numpy import typing as npt
from astropy import cosmology as cosmo
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from supaernova.utils import resolve_path
from supaernova.configs.steps import StepConfig, StepResult, StepAnalysis
from supaernova.analysis.spectra import SpectraPlot, ResidualPlot
from supaernova.configs.steps.variants import VariantConfig


class SNPAEData(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    ind: "npt.NDArray[np.int32]"
    nspectra: "npt.NDArray[np.int32]"
    sn_name: "npt.NDArray[np.str_]"
    dphase: "npt.NDArray[np.float32]"
    redshift: "npt.NDArray[np.float32]"
    x0: "npt.NDArray[np.float32]"
    x1: "npt.NDArray[np.float32]"
    c: "npt.NDArray[np.float32]"
    mb: "npt.NDArray[np.float32]"
    hubble_residual: "npt.NDArray[np.float32]"
    luminosity_distance: "npt.NDArray[np.float32]"
    spectra_id: "npt.NDArray[np.str_]"
    phase: "npt.NDArray[np.float32]"
    wl_mask_min: "npt.NDArray[np.float32]"
    wl_mask_max: "npt.NDArray[np.float32]"
    amplitude: "npt.NDArray[np.float32]"
    sigma: "npt.NDArray[np.float32]"
    salt_flux: "npt.NDArray[np.float32]"
    wavelength: "npt.NDArray[np.float32]"
    mask: "npt.NDArray[np.int32]"
    sn_mask: "npt.NDArray[np.int32]"
    spec_mask: "npt.NDArray[np.int32]"
    wl_mask: "npt.NDArray[np.int32]"
    time: "npt.NDArray[np.float32]"
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class DataStepResult(StepResult):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    data: SNPAEData
    train_data: list[SNPAEData]
    test_data: list[SNPAEData]
    # --- Optional ---
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class DataStepAnalysis(StepAnalysis):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    plot_spectra: SpectraPlot | list[SpectraPlot] | None = None
    plot_summary: SpectraPlot | list[SpectraPlot] | None = None
    plot_residual: ResidualPlot | list[ResidualPlot] | None = None
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


class DataConfig(StepConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    data_dir: Path
    meta: Path
    idr: Path
    mask: Path
    train_frac: Annotated[float, Field(ge=0, le=1)]

    # --- Optional ---
    colourlaw: Path | None = None
    analysis: DataStepAnalysis | None = None
    cosmological_model: str = "WMAP7"
    salt_model: str | Path = "salt2"
    min_redshift: float | Literal["inf", "-inf"] = -np.inf
    max_redshift: float | Literal["inf", "-inf"] = np.inf
    min_phase: float | Literal["inf", "-inf"] = -np.inf
    max_phase: float | Literal["inf", "-inf"] = np.inf
    min_wavelength: float | Literal["inf", "-inf"] = -np.inf
    max_wavelength: float | Literal["inf", "-inf"] = np.inf

    # === Model Validators ===
    # --- Before ---
    @classmethod
    @model_validator(mode="before")
    def _validate_bounds(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for var in ["redshift", "phase", "wavelength"]:
                min_bound = data.get(f"min_{var}")
                if min_bound == "inf":
                    min_bound = np.inf
                elif min_bound == "-inf":
                    min_bound = -np.inf
                max_bound = data.get(f"max_{var}")
                if max_bound == "inf":
                    max_bound = np.inf
                elif max_bound == "-inf":
                    max_bound = -np.inf
                if max_bound <= min_bound:
                    err = f"`max_{var}`: {max_bound} is not strictly greater than `min_{var}`: {min_bound}"
                    cls._raise(err)
                data[f"min_{var}"] = min_bound
                data[f"max_{var}"] = max_bound
        return data

    # --- After ---
    @model_validator(mode="after")
    def _validate_paths(self) -> Self:
        self.data_dir = resolve_path(self.data_dir, relative_path=self.paths.base)
        if not self.data_dir.exists():
            err = f"`data_dir` resolved to {self.data_dir}, which does not exist."
            self._raise(err)

        for field, ext in {"meta": ".csv", "idr": ".txt", "mask": ".txt"}.items():
            setattr(
                self,
                field,
                resolve_path(getattr(self, field), relative_path=self.data_dir),
            )

            field_path: Path = getattr(self, field)

            if not field_path.exists():
                err = f"`{field}` resolved to {field_path}, which does not exist."
                self._raise(err)

            if field_path.suffix != ext:
                err = f"`{field}` resolved to {field_path}, which is not a {ext} file."
                self._raise(err)

        if self.colourlaw is not None:
            self.colourlaw = resolve_path(self.colourlaw, relative_path=self.data_dir)
            if not self.colourlaw.exists():
                err = f"`colourlaw` resolved to {self.colourlaw}, which does not exist."
                self._raise(err)
        return self

    @model_validator(mode="after")
    def _validate_salt_model_path(self) -> Self:
        salt_path = resolve_path(Path(self.salt_model), relative_path=self.paths.base)
        if salt_path.exists():
            self.salt_model = salt_path
        return self

    # === Field Validators ===
    # --- Before ---
    # --- After ---
    @field_validator("cosmological_model", mode="after")
    @classmethod
    def _validate_cosmological_model(cls, value: str) -> str:
        if value not in cosmo.realizations.available:
            err = f"`cosmological_model` is {value} but must be one of {cosmo.realizations.available}"
            cls._raise(err)
        return value

    @field_validator("salt_model", mode="after")
    @classmethod
    def _validate_salt_model(cls, value: str) -> str:
        if ("salt2" not in value) and ("salt3" not in value):
            err = f'`salt_model` is {value} but does not appear to be a salt2 or salt3 model, as it does not contain the string `"salt2"` or `"salt3"'
            cls._raise(err)
        return value

    # === Instance Methods ===
    # === Static Methods ===


class DataStepConfig(VariantConfig):
    # === Class Variables ===
    id: ClassVar[str] = "data"
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    base: DataConfig
    # --- Optional ---
    variants: list[DataConfig] | None = Field(None, validation_alias="variant")
    # === Model Validators ===
    # --- Before ---
    # --- After ---
    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    # === Static Methods ===


DataStepConfig.register_step(DataConfig)
