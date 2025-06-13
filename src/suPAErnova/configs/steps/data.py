# Copyright 2025 Patrick Armstrong


from typing import (
    Self,
    ClassVar,
    Annotated,
)
from pathlib import Path

import numpy as np
from numpy import typing as npt
from astropy import cosmology as cosmo
from pydantic import (
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

from .steps import StepConfig, AbstractStepResult


class DataStepResult(AbstractStepResult):
    ind: "npt.NDArray[np.int32]"
    nspectra: "npt.NDArray[np.int32]"
    sn_name: "npt.NDArray[np.str_]"
    dphase: "npt.NDArray[np.float32]"
    redshift: "npt.NDArray[np.float32]"
    x0: "npt.NDArray[np.float32]"
    x1: "npt.NDArray[np.float32]"
    c: "npt.NDArray[np.float32]"
    MB: "npt.NDArray[np.float32]"
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
    time: "npt.NDArray[np.float32]"


class DataStepConfig(StepConfig):
    # --- Class Variables ---
    id: ClassVar[str] = "data"

    # --- Required ---
    data_dir: Path
    meta: Path
    idr: Path
    mask: Path

    # --- Optional ---
    cosmological_model: str
    salt_model: str | Path
    min_phase: float
    max_phase: float
    train_frac: Annotated[float, Field(ge=0, le=1)]
    seed: PositiveInt

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        self.data_dir = self.paths.resolve_path(
            self.data_dir, relative_path=self.paths.base
        )
        if not self.data_dir.exists():
            err = f"`data_dir` resolved to {self.data_dir}, which does not exist."
            self._raise(err)

        for field, ext in {"meta": ".csv", "idr": ".txt", "mask": ".txt"}.items():
            setattr(
                self,
                field,
                self.paths.resolve_path(
                    getattr(self, field), relative_path=self.data_dir
                ),
            )

            field_path: Path = getattr(self, field)

            if not field_path.exists():
                err = f"`{field}` resolved to {field_path}, which does not exist."
                self._raise(err)

            if field_path.suffix != ext:
                err = f"`{field}` resolved to {field_path}, which is not a {ext} file."
                self._raise(err)

        return self

    @field_validator("cosmological_model", mode="after")
    @classmethod
    def validate_cosmological_model(cls, value: str) -> str:
        if value not in cosmo.realizations.available:
            err = f"`cosmological_model` is {value} but must be one of {cosmo.realizations.available}"
            cls._raise(err)
        return value

    @field_validator("salt_model", mode="after")
    @classmethod
    def validate_salt_model(cls, value: str) -> str:
        if ("salt2" not in value) and ("salt3" not in value):
            err = f'`salt_model` is {value} but does not appear to be a salt2 or salt3 model, as it does not contain the string `"salt2"` or `"salt3"'
            cls._raise(err)
        return value

    @model_validator(mode="after")
    def validate_salt_model_path(self) -> Self:
        salt_path = self.paths.resolve_path(
            Path(self.salt_model), relative_path=self.paths.base
        )
        if salt_path.exists():
            self.salt_model = salt_path
        return self

    @model_validator(mode="after")
    def validate_max_phase(self) -> Self:
        if self.max_phase <= self.min_phase:
            err = f"`max_phase`: {self.max_phase} is not strictly greater than `min_phase`: {self.min_phase}"
            self._raise(err)
        return self


DataStepConfig.register_step()
