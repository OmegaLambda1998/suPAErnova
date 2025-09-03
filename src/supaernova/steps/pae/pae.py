# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, Any, Self, override
from pathlib import Path
import importlib

import numpy as np

from supaernova.analysis import Plotter
from supaernova.steps.models import Model, ModelStep
from supaernova.analysis.spectra import SpectraPlotter
from supaernova.configs.callbacks import callback
from supaernova.configs.steps.pae import (
    PAEStage,
    PAEConfig,
    PAEStepConfig,
    PAEStepResult,
    PAEStageResult,
    PAEStepAnalysis,
)
from supaernova.analysis.distribution import DistributionPlotter

if TYPE_CHECKING:
    from typing import ClassVar
    from collections.abc import Callable

    from numpy import typing as npt

    from supaernova.configs.steps.data import LazySNPAEData, DataStepResult

    from .tf import TFPAEModel

    PAEModel = TFPAEModel


class PAE(ModelStep[PAEConfig]):
    def __init__(self: Self, config: PAEConfig) -> None:
        super().__init__(config)

        # === Config Variables ===
        # --- Required ---
        self.physical_latents: bool = self.options.physical_latents
        self.n_physical_latents = 3 if self.physical_latents else 0
        self.seperate_latent_training: bool = self.options.seperate_latent_training
        self.seperate_z_latent_training: bool = self.options.seperate_z_latent_training
        self.validation_frac: float = self.options.validation_frac

        # --- Optional ---
        self.debug: bool = self.options.debug
        self.profile: bool = self.options.profile
        self.n_z_latents: int = self.options.n_z_latents
        self.n_pae_latents = self.n_physical_latents + self.n_z_latents
        self.kfold: int = self.options.kfold

        # === Setup Variables ===
        self.setup_attributes: set[str] = {
            "data",
            "mask",
            "sn_mask",
            "spec_mask",
            "wl_mask",
            "train_data",
            "train_mask",
            "train_sn_mask",
            "train_spec_mask",
            "train_wl_mask",
            "test_data",
            "test_mask",
            "test_sn_mask",
            "test_spec_mask",
            "test_wl_mask",
            "val_data",
            "val_mask",
            "val_sn_mask",
            "val_spec_mask",
            "val_wl_mask",
            "colourlaw",
            "min_redshift",
            "max_redshift",
            "min_train_redshift",
            "max_train_redshift",
            "min_test_redshift",
            "max_test_redshift",
            "min_val_redshift",
            "max_val_redshift",
            "min_phase",
            "max_phase",
            "min_train_phase",
            "max_train_phase",
            "min_test_phase",
            "max_test_phase",
            "min_val_phase",
            "max_val_phase",
            "min_wavelength",
            "max_wavelength",
            "min_train_wavelength",
            "max_train_wavelength",
            "min_test_wavelength",
            "max_test_wavelength",
            "min_val_wavelength",
            "max_val_wavelength",
            "batch_size",
            "sn_dim",
            "spec_dim",
            "wl_dim",
            "stage_delta_av",
            "stage_zs",
            "stage_delta_m",
            "stage_delta_p",
            "stage_final",
            "run_stages",
            "model",
        }

        # --- Previous Step Variables ---
        self.data: LazySNPAEData
        self.mask: npt.NDArray[bool]
        self.sn_mask: npt.NDArray[bool]
        self.spec_mask: npt.NDArray[bool]
        self.wl_mask: npt.NDArray[bool]

        self.train_data: LazySNPAEData
        self.train_mask: npt.NDArray[bool]
        self.train_sn_mask: npt.NDArray[bool]
        self.train_spec_mask: npt.NDArray[bool]
        self.train_wl_mask: npt.NDArray[bool]

        self.test_data: LazySNPAEData
        self.test_mask: npt.NDArray[bool]
        self.test_sn_mask: npt.NDArray[bool]
        self.test_spec_mask: npt.NDArray[bool]
        self.test_wl_mask: npt.NDArray[bool]

        self.val_data: LazySNPAEData
        self.val_mask: npt.NDArray[bool]
        self.val_sn_mask: npt.NDArray[bool]
        self.val_spec_mask: npt.NDArray[bool]
        self.val_wl_mask: npt.NDArray[bool]

        self.colourlaw: npt.NDArray[float] | None

        # --- Bounds ---
        self.min_redshift: float
        self.max_redshift: float
        self.min_train_redshift: float
        self.max_train_redshift: float
        self.min_test_redshift: float
        self.max_test_redshift: float
        self.min_val_redshift: float
        self.max_val_redshift: float

        self.min_phase: float
        self.max_phase: float
        self.min_train_phase: float
        self.max_train_phase: float
        self.min_test_phase: float
        self.max_test_phase: float
        self.min_val_phase: float
        self.max_val_phase: float

        self.min_wavelength: float
        self.max_wavelength: float
        self.min_train_wavelength: float
        self.max_train_wavelength: float
        self.min_test_wavelength: float
        self.max_test_wavelength: float
        self.min_val_wavelength: float
        self.max_val_wavelength: float

        self.batch_size: int

        self.model: PAEModel

        # Data Dimensions
        self.sn_dim: int
        self.spec_dim: int
        self.wl_dim: int
        self.phase_dim: int = 1

        # PAE Stages
        self.stage_delta_av: PAEStage
        self.stage_zs: list[PAEStage]
        self.stage_delta_m: PAEStage
        self.stage_delta_p: PAEStage
        self.stage_final: PAEStage
        self.run_stages: list[PAEStage]

        # === Run / Save / Load Variables ===
        self._run_flag: bool
        self.run_attributes: set[str] = {"_run_flag"}
        self.save_attributes: set[str] = self.run_attributes
        self.load_attributes: set[str] = self.save_attributes

        # === Result Variables ===
        self.results: PAEStepResult

        # === Analysis Variables ===
        self.analysis: PAEStepAnalysis = self.options.analysis or PAEStepAnalysis()

    @override
    def _is_setup(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(self.setup_attributes)

    @override
    def _setup(self: Self, *args: Any, data: "DataStepResult", **kwargs: Any) -> None:
        super()._setup()
        # --- Previous Step Variables ---
        self.data = data.data
        self.colourlaw = data.colourlaw
        self.train_data = data.train_data[self.kfold % len(data.train_data)]
        self.test_data = data.test_data[self.kfold % len(data.test_data)]
        self.val_data = self.test_data
        if self.validation_frac > 0:
            ind_split = int(data.sn_dim * self.validation_frac)
            self.val_data.model_validate({
                k: v[-ind_split:] for k, v in self.train_data.model_dump().items()
            })
            self.train_data.model_validate({
                k: v[:-ind_split] for k, v in self.train_data.model_dump().items()
            })

        n_batches = self.options.n_batches
        self.batch_size = max(int(data.train_frac * data.sn_dim / n_batches), 1)

        # --- Bounds ---
        self.min_redshift = self.options.min_redshift or data.min_redshift
        self.max_redshift = self.options.max_redshift or data.max_redshift
        self.min_train_redshift = self.options.min_train_redshift or self.min_redshift
        self.max_train_redshift = self.options.max_train_redshift or self.max_redshift
        self.min_test_redshift = self.options.min_test_redshift or self.min_redshift
        self.max_test_redshift = self.options.max_test_redshift or self.max_redshift
        self.min_val_redshift = self.options.min_val_redshift or self.min_redshift
        self.max_val_redshift = self.options.max_val_redshift or self.max_redshift

        self.min_phase = self.options.min_phase or data.min_phase
        self.max_phase = self.options.max_phase or data.max_phase
        self.min_train_phase = self.options.min_train_phase or self.min_phase
        self.max_train_phase = self.options.max_train_phase or self.max_phase
        self.min_test_phase = self.options.min_test_phase or self.min_phase
        self.max_test_phase = self.options.max_test_phase or self.max_phase
        self.min_val_phase = self.options.min_val_phase or self.min_phase
        self.max_val_phase = self.options.max_val_phase or self.max_phase

        self.min_wavelength = self.options.min_wavelength or data.min_wavelength
        self.max_wavelength = self.options.max_wavelength or data.max_wavelength
        self.min_train_wavelength = (
            self.options.min_train_wavelength or self.min_wavelength
        )
        self.max_train_wavelength = (
            self.options.max_train_wavelength or self.max_wavelength
        )
        self.min_test_wavelength = (
            self.options.min_test_wavelength or self.min_wavelength
        )
        self.max_test_wavelength = (
            self.options.max_test_wavelength or self.max_wavelength
        )
        self.min_val_wavelength = self.options.min_val_wavelength or self.min_wavelength
        self.max_val_wavelength = self.options.max_val_wavelength or self.max_wavelength

        self.setup_data_masks()

        # Data Dimensions
        self.sn_dim = data.sn_dim
        self.spec_dim = data.spec_dim
        self.wl_dim = data.wl_dim

        # PAE Stages
        stage_data = {
            "data": self.data,
            "mask": self.mask,
            "sn_mask": self.sn_mask,
            "spec_mask": self.spec_mask,
            "wl_mask": self.wl_mask,
            "train_data": self.train_data,
            "train_mask": self.train_mask,
            "train_sn_mask": self.train_sn_mask,
            "train_spec_mask": self.train_spec_mask,
            "train_wl_mask": self.train_wl_mask,
            "test_data": self.test_data,
            "test_mask": self.test_mask,
            "test_sn_mask": self.test_sn_mask,
            "test_spec_mask": self.test_spec_mask,
            "test_wl_mask": self.test_wl_mask,
            "val_data": self.val_data,
            "val_mask": self.val_mask,
            "val_sn_mask": self.val_sn_mask,
            "val_spec_mask": self.val_spec_mask,
            "val_wl_mask": self.val_wl_mask,
            "debug": self.debug,
            "profile": self.profile,
        }

        self.stage_delta_av = PAEStage.model_validate({
            "stage": 1,
            "prev_stage": None,
            "name": "ΔAᵥ",
            "fname": "delta_av",
            "epochs": self.options.delta_av_epochs,
            "patience": self.options.delta_av_patience,
            "learning_rate": self.options.delta_av_lr,
            "learning_rate_decay_steps": self.options.delta_av_lr_decay_steps,
            "learning_rate_decay_rate": self.options.delta_av_lr_decay_rate,
            "learning_rate_weight_decay_rate": self.options.delta_av_lr_weight_decay_rate,
            **stage_data,
        })

        z0 = 2 if self.physical_latents else 1
        self.stage_zs = [
            PAEStage.model_validate({
                "stage": z0 + i,
                "prev_stage": z0 + i - 1,
                "name": f"z{i + 1}",
                "fname": f"z{i + 1}",
                "epochs": self.options.zs_epochs,
                "patience": self.options.zs_patience,
                "learning_rate": self.options.zs_lr,
                "learning_rate_decay_steps": self.options.zs_lr_decay_steps,
                "learning_rate_decay_rate": self.options.zs_lr_decay_rate,
                "learning_rate_weight_decay_rate": self.options.zs_lr_weight_decay_rate,
                **stage_data,
            })
            for i in range(self.n_z_latents)
        ]
        if not self.seperate_z_latent_training:
            self.stage_zs = self.stage_zs[-1:]
            self.stage_zs[0].prev_stage = 1
            self.stage_zs[0].name = "zs"

        self.stage_delta_m = PAEStage.model_validate({
            "stage": z0 + self.n_z_latents,
            "prev_stage": z0 + self.n_z_latents - 1,
            "name": "Δℳ",
            "fname": "delta_m",
            "epochs": self.options.delta_m_epochs,
            "patience": self.options.delta_m_patience,
            "learning_rate": self.options.delta_m_lr,
            "learning_rate_decay_steps": self.options.delta_m_lr_decay_steps,
            "learning_rate_decay_rate": self.options.delta_m_lr_decay_rate,
            "learning_rate_weight_decay_rate": self.options.delta_m_lr_weight_decay_rate,
            **stage_data,
        })

        self.stage_delta_p = PAEStage.model_validate({
            "stage": z0 + self.n_z_latents + 1,
            "prev_stage": z0 + self.n_z_latents,
            "name": "Δp",
            "fname": "delta_p",
            "epochs": self.options.delta_p_epochs,
            "patience": self.options.delta_p_patience,
            "learning_rate": self.options.delta_p_lr,
            "learning_rate_decay_steps": self.options.delta_p_lr_decay_steps,
            "learning_rate_decay_rate": self.options.delta_p_lr_decay_rate,
            "learning_rate_weight_decay_rate": self.options.delta_p_lr_weight_decay_rate,
            **stage_data,
        })

        self.stage_final = PAEStage.model_validate({
            "stage": self.n_pae_latents,
            "prev_stage": None,
            "name": "Final",
            "fname": "final",
            "epochs": self.options.final_epochs,
            "patience": self.options.final_patience,
            "learning_rate": self.options.final_lr,
            "learning_rate_decay_steps": self.options.final_lr_decay_steps,
            "learning_rate_decay_rate": self.options.final_lr_decay_rate,
            "learning_rate_weight_decay_rate": self.options.final_lr_weight_decay_rate,
            **stage_data,
        })

        if self.physical_latents:
            self.run_stages = [
                self.stage_delta_av,
                *self.stage_zs,
                self.stage_delta_m,
                self.stage_delta_p,
            ]
        else:
            self.run_stages = self.stage_zs

        if not self.seperate_latent_training:
            self.run_stages = [self.stage_final]

    @override
    def _has_run(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(self.run_attributes)

    @override
    def _run(self: Self, *args: Any, **kwargs: Any) -> None:
        savepath: Path | None = None
        for i, stage in enumerate(self.run_stages):
            self.model = self.model.__class__(self)
            self.log.debug(f"Starting PAEStage {i}: {stage.name}")
            if savepath is not None:
                stage.loadpath = savepath
            savepath = (
                self.paths.results / self.model.name / f"{stage.stage}_{stage.fname}"
            )
            stage.savepath = savepath

            ckpt_path = savepath / self.model.ckpt_path
            # Don't retrain stages if you don't need to
            if self.force or not (ckpt_path.exists() and any(ckpt_path.iterdir())):
                self.model.train_model(stage)
                self.model.save_checkpoint(savepath)

        self._run_flag = True

    @override
    def _is_saved(self: Self, *args: Any, **kwargs: Any) -> bool:
        final_savepath = self.paths.results / self.model.name / self.model.ckpt_path
        if not (final_savepath.exists() and any(final_savepath.iterdir())):
            self.log.debug(
                f"{self.name} is not saved as {final_savepath} does not exist"
            )
            return False
        return True

    @override
    def _save(self: Self, *args: "Any", **kwargs: "Any") -> None:
        self.model = self.model.__class__(self)
        final_savepath = self.paths.results / self.model.name
        final_stage = self.run_stages[-1]
        final_stage.prev_stage = None
        self.model.stage = final_stage
        savepath = (
            self.paths.results
            / self.model.name
            / f"{final_stage.stage}_{final_stage.fname}"
        )
        self.model.load_checkpoint(savepath, reset_weights=False)
        self.log.debug(f"Saving final PAE model weights to {final_savepath}")
        self.model.save_checkpoint(final_savepath)

    @override
    def _load(self: Self, *args: Any, **kwargs: Any) -> None:
        self.model = self.model.__class__(self)
        final_stage = self.run_stages[-1]
        final_stage.prev_stage = None
        self.model.stage = final_stage
        final_loadpath = self.paths.results / self.model.name
        self.log.debug(f"Loading final PAE model weights from {final_loadpath}")
        self.model.load_checkpoint(final_loadpath, reset_weights=False)
        self._run_flag = True

    @override
    def _has_results(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        return self.has_attributes(["results"])

    @override
    def _result(self: Self, *args: Any, **kwargs: Any) -> None:
        pae_results = {}
        pae_results["min_redshift"] = self.min_redshift
        pae_results["max_redshift"] = self.max_redshift
        pae_results["min_phase"] = self.min_phase
        pae_results["max_phase"] = self.max_phase
        pae_results["min_wavelength"] = self.min_wavelength
        pae_results["max_wavelength"] = self.max_wavelength

        self._load(*args, **kwargs)

        pae_results["model"] = self.model

        self.log.debug("Calculating PAE results")
        dt_results: dict[str, dict[str, PAEStageResult]] = {}
        for dt in ["train_", "test_"]:
            model_results: dict[str, PAEStageResult] = {}
            data = getattr(self, f"{dt}data")
            input_ind = data.ind
            input_sn_name = data.sn_name
            input_spectra_id = data.spectra_id
            input_phase = data.time
            input_amplitude = data.amplitude
            input_d_amplitude = data.sigma
            data.clear()
            input_mask = getattr(self, f"{dt}mask")
            input_sn_mask = getattr(self, f"{dt}sn_mask")
            input_spec_mask = getattr(self, f"{dt}spec_mask")
            input_wl_mask = getattr(self, f"{dt}wl_mask")
            for stage in self.run_stages:
                self.model = self.model.__class__(self)
                savepath = (
                    self.paths.results
                    / self.model.name
                    / f"{stage.stage}_{stage.fname}"
                )
                stage.prev_stage = None
                self.model.stage = stage
                self.model.load_checkpoint(savepath, reset_weights=False)

                latents, output_amplitude = self.model(
                    (input_phase, input_amplitude),
                    training=False,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                )

                loss = self.model.compute_loss(
                    latents,
                    input_amplitude,
                    output_amplitude,
                    sample_weight=input_d_amplitude,
                    training=False,
                    mask=input_mask,
                    sn_mask=input_sn_mask,
                    spec_mask=input_spec_mask,
                    wl_mask=input_wl_mask,
                )

                pred_loss = self.model.get_loss("loss_pred")
                model_loss = self.model.get_loss("loss_model")
                resid_loss = self.model.get_loss("loss_resid")
                delta_loss = self.model.get_loss("loss_delta")
                cov_loss = self.model.get_loss("loss_cov")

                results = {
                    "stage": stage.stage,
                    "ind": input_ind,
                    "sn_name": input_sn_name,
                    "spectra_id": input_spectra_id,
                    "input_amp": input_amplitude,
                    "input_d_amp": input_d_amplitude,
                    "input_phase": input_phase,
                    "input_mask": np.array(input_mask),
                    "input_sn_mask": np.array(input_sn_mask),
                    "input_spec_mask": np.array(input_spec_mask),
                    "input_wl_mask": np.array(input_wl_mask),
                    "input_colourlaw": self.model.decoder.colourlaw,
                    "latents": latents.numpy()[:, 0, :],
                    "output_amp": output_amplitude.numpy(),
                    "diff_amp": np.abs(input_amplitude - output_amplitude.numpy()),
                    "loss": loss,
                    "pred_loss": pred_loss,
                    "model_loss": model_loss,
                    "resid_loss": resid_loss,
                    "delta_loss": delta_loss,
                    "cov_loss": cov_loss,
                }
                stage_results = PAEStageResult.model_validate(results)
                model_results[str(stage.stage)] = stage_results
            dt_results[dt[:-1]] = model_results

        pae_results["stages"] = dt_results

        self.results = PAEStepResult.model_validate(pae_results)

    @override
    def _was_analysed(self: Self, *args: "Any", **kwargs: "Any") -> bool:
        for dt in ["train_", "test_"]:
            for stage in self.run_stages:
                if self.analysis.plot_comparison is not None:
                    if not isinstance(self.analysis.plot_comparison, list):
                        self.analysis.plot_comparison = [self.analysis.plot_comparison]
                    for opts in self.analysis.plot_comparison:
                        name = "comparison" if opts.name is None else opts.name
                        savepath = (
                            self.paths.plots
                            / dt[:-1]
                            / str(self.model.seed)
                            / str(stage.stage)
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False

                if self.analysis.plot_latents is not None:
                    if not isinstance(self.analysis.plot_latents, list):
                        self.analysis.plot_latents = [self.analysis.plot_latents]
                    for opts in self.analysis.plot_latents:
                        name = "latents" if opts.name is None else opts.name
                        savepath = (
                            self.paths.plots
                            / dt[:-1]
                            / str(self.model.seed)
                            / str(stage.stage)
                            / f"{name}.{opts.ext}"
                            if opts.savepath is None
                            else opts.savepath
                        )
                        if not savepath.exists():
                            self.log.debug(
                                f"{self.name} is missing analyses as {savepath} does not exist"
                            )
                            return False

            if self.analysis.plot_comparison is not None:
                if not isinstance(self.analysis.plot_comparison, list):
                    self.analysis.plot_comparison = [self.analysis.plot_comparison]
                for opts in self.analysis.plot_comparison:
                    name = "comparison" if opts.name is None else opts.name
                    savepath = (
                        self.paths.plots
                        / dt[:-1]
                        / str(self.model.seed)
                        / f"{name}.{opts.ext}"
                        if opts.savepath is None
                        else opts.savepath
                    )
                    if not savepath.exists():
                        self.log.debug(
                            f"{self.name} is missing analyses as {savepath} does not exist"
                        )
                        return False

            if self.analysis.plot_latents is not None:
                if not isinstance(self.analysis.plot_latents, list):
                    self.analysis.plot_latents = [self.analysis.plot_latents]
                for opts in self.analysis.plot_latents:
                    name = "latents" if opts.name is None else opts.name
                    savepath = (
                        self.paths.plots
                        / dt[:-1]
                        / str(self.model.seed)
                        / f"{name}.{opts.ext}"
                        if opts.savepath is None
                        else opts.savepath
                    )
                    if not savepath.exists():
                        self.log.debug(
                            f"{self.name} is missing analyses as {savepath} does not exist"
                        )
                        return False

        return True

    @override
    def _analyse(self: Self, *args: Any, **kwargs: Any) -> None:
        labels = {}
        ind = 0
        if self.physical_latents:
            labels[0] = "ΔAᵥ"
            ind = 1
            labels[self.n_pae_latents - 2] = "Δℳ"
            labels[self.n_pae_latents - 1] = "Δp"
        for i in range(self.n_z_latents):
            labels[ind] = f"z{i + 1}"
            ind += 1

        for dt in ["train_", "test_"]:
            for stage in self.run_stages:
                results = self.results.stages[dt[:-1]][str(stage.stage)]

                input_mask = results.input_mask
                input_sn_mask = results.input_sn_mask
                input_spec_mask = results.input_spec_mask
                input_wl_mask = results.input_wl_mask

                if self.analysis.plot_comparison is not None:
                    if not isinstance(self.analysis.plot_comparison, list):
                        self.analysis.plot_comparison = [self.analysis.plot_comparison]
                    for opts in self.analysis.plot_comparison:
                        o = opts.model_copy(deep=True)
                        if o.name is None:
                            o.name = "comparison"
                        self.log.debug(f"Plotting {o.name}")
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
                                / dt[:-1]
                                / str(self.model.seed)
                                / str(stage.stage)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        if o.plot_kwargs is None:
                            o.plot_kwargs = {
                                "label": f"{dt}{self.name}_{stage.name}\n({results.loss:.2E})",
                                "title": f"{dt}{self.name}_{stage.name} {o.name}",
                            }

                        data = getattr(self, f"{dt}data")
                        (
                            wl,
                            amplitude,
                            sigma,
                            _sn_name,
                            _time,
                            mask,
                            _sn_mask,
                            _spec_mask,
                            _wl_mask,
                        ) = SpectraPlotter.prep(
                            data,
                            o,
                            mask=input_mask,
                            sn_mask=input_sn_mask,
                            spec_mask=input_spec_mask,
                            wl_mask=input_wl_mask,
                        )
                        o.base_wl = wl
                        o.base_amp = amplitude
                        o.base_sigma = sigma
                        o.base_mask = np.logical_not(mask)

                        data.amplitude = results.output_amp
                        data.sigma *= 0

                        SpectraPlotter.plot_comparison(
                            data,
                            o,
                            mask=input_mask,
                            sn_mask=input_sn_mask,
                            spec_mask=input_spec_mask,
                            wl_mask=input_wl_mask,
                        )

                if self.analysis.plot_latents is not None:
                    if not isinstance(self.analysis.plot_latents, list):
                        self.analysis.plot_latents = [self.analysis.plot_latents]
                    for opts in self.analysis.plot_latents:
                        o = opts.model_copy(deep=True)
                        if o.labels is None:
                            o.labels = {i: labels[i] for i in range(stage.stage)}
                        if o.name is None:
                            o.name = "latents"
                        self.log.debug(f"Plotting {o.name}")
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
                                / dt[:-1]
                                / str(self.model.seed)
                                / str(stage.stage)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        if o.plot_kwargs is None:
                            o.plot_kwargs = {"title": f"{dt}{self.name}_{stage.name}"}

                        data = getattr(self, f"{dt}data")
                        (
                            wl,
                            amplitude,
                            sigma,
                            _sn_name,
                            _time,
                            mask,
                            _sn_mask,
                            _spec_mask,
                            _wl_mask,
                        ) = SpectraPlotter.prep(
                            data,
                            o,
                            mask=input_mask,
                            sn_mask=input_sn_mask,
                            spec_mask=input_spec_mask,
                            wl_mask=input_wl_mask,
                        )

                        # ~(~input_mask & input_wl_mask)
                        # Extracts unmasked wavelengths from the valid wavelength range provided by wl_mask
                        valid_wl_mask = np.logical_not(
                            np.logical_and(np.logical_not(input_mask), input_wl_mask)
                        )

                        # Determine which spectra to keep
                        # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
                        mask_spec = np.min(valid_wl_mask, axis=-1, keepdims=True)

                        # Determine which SNe to keep
                        # Will mask out any SN with *no* unmasked spectra
                        mask_sn = np.max(mask_spec, axis=-2)[:, 0]

                        chains = results.latents[:, : stage.stage][mask_sn]

                        DistributionPlotter.plot_corner(
                            chains,
                            o,
                            statistics="max_central",
                            shade_alpha=0.0,
                            plot_cloud=True,
                            smooth=0,
                            bins=self.sn_dim,
                        )

            if self.analysis.plot_comparison is not None:
                results = self.results.stages[dt[:-1]][str(self.run_stages[0].stage)]
                if not isinstance(self.analysis.plot_comparison, list):
                    self.analysis.plot_comparison = [self.analysis.plot_comparison]
                for opts in self.analysis.plot_comparison:
                    o = opts.model_copy(deep=True)
                    if o.name is None:
                        o.name = "comparison"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt[:-1] / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {}
                    o.plot_kwargs["label"] = (
                        f"{dt}{self.name}_{self.run_stages[0].name}\n({results.loss:.2E})",
                    )
                    o.plot_kwargs["title"] = f"{dt}{self.name} {o.name}"
                    savepath = (o.savepath or Path()) / f"{o.name}.{o.ext}"
                    if not savepath.exists():
                        input_mask = results.input_mask
                        input_sn_mask = results.input_sn_mask
                        input_spec_mask = results.input_spec_mask
                        input_wl_mask = results.input_wl_mask

                        data = getattr(self, f"{dt}data")
                        (
                            wl,
                            amplitude,
                            sigma,
                            _sn_name,
                            _time,
                            mask,
                            _sn_mask,
                            _spec_mask,
                            _wl_mask,
                        ) = SpectraPlotter.prep(
                            data,
                            o,
                            mask=input_mask,
                            sn_mask=input_sn_mask,
                            spec_mask=input_spec_mask,
                            wl_mask=input_wl_mask,
                        )
                        o.base_wl = wl
                        o.base_amp = amplitude
                        o.base_sigma = sigma
                        o.base_mask = np.logical_not(mask)

                        data.amplitude = results.output_amp
                        data.sigma *= 0

                        fig, ax = SpectraPlotter.plot_comparison(
                            data,
                            o,
                            save=False,
                            mask=input_mask,
                            sn_mask=input_sn_mask,
                            spec_mask=input_spec_mask,
                            wl_mask=input_wl_mask,
                        )
                        for stage in self.run_stages[1:]:
                            results = self.results.stages[dt[:-1]][str(stage.stage)]
                            input_mask = results.input_mask
                            input_sn_mask = results.input_sn_mask
                            input_spec_mask = results.input_spec_mask
                            input_wl_mask = results.input_wl_mask

                            data = getattr(self, f"{dt}data")
                            (
                                wl,
                                amplitude,
                                sigma,
                                _sn_name,
                                _time,
                                mask,
                                _sn_mask,
                                _spec_mask,
                                _wl_mask,
                            ) = SpectraPlotter.prep(
                                data,
                                o,
                                mask=input_mask,
                                sn_mask=input_sn_mask,
                                spec_mask=input_spec_mask,
                                wl_mask=input_wl_mask,
                            )
                            o.base_wl = wl
                            o.base_amp = amplitude
                            o.base_sigma = sigma
                            o.base_mask = np.logical_not(mask)

                            data.amplitude = results.output_amp
                            data.sigma *= 0

                            o.plot_kwargs["label"] = (
                                f"{dt}{self.name}_{stage.name}\n({results.loss:.2E})"
                            )
                            o.plot_base = False

                            fig, ax = SpectraPlotter.plot_comparison(
                                data,
                                o,
                                fig=fig,
                                ax=ax,
                                save=False,
                                mask=input_mask,
                                sn_mask=input_sn_mask,
                                spec_mask=input_spec_mask,
                                wl_mask=input_wl_mask,
                            )

                        fig = Plotter.save(fig, savepath)
                        Plotter.close(fig, ax)

            if self.analysis.plot_latents is not None:
                results = self.results.stages[dt[:-1]][str(self.run_stages[0].stage)]
                if not isinstance(self.analysis.plot_latents, list):
                    self.analysis.plot_latents = [self.analysis.plot_latents]
                for opts in self.analysis.plot_latents:
                    o = opts.model_copy(deep=True)
                    if o.labels is None:
                        o.labels = {
                            stage.name: {i: labels[i] for i in range(stage.stage)}
                            for stage in self.run_stages
                        }
                    if o.name is None:
                        o.name = "latents"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt[:-1] / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)
                    if o.plot_kwargs is None:
                        o.plot_kwargs = {"title": f"{dt}{self.name}"}
                    input_mask = results.input_mask
                    input_sn_mask = results.input_sn_mask
                    input_spec_mask = results.input_spec_mask
                    input_wl_mask = results.input_wl_mask

                    data = getattr(self, f"{dt}data")
                    (
                        wl,
                        amplitude,
                        sigma,
                        _sn_name,
                        _time,
                        mask,
                        _sn_mask,
                        _spec_mask,
                        _wl_mask,
                    ) = SpectraPlotter.prep(
                        data,
                        o,
                        mask=input_mask,
                        sn_mask=input_sn_mask,
                        spec_mask=input_spec_mask,
                        wl_mask=input_wl_mask,
                    )

                    # ~(~input_mask & input_wl_mask)
                    # Extracts unmasked wavelengths from the valid wavelength range provided by wl_mask
                    valid_wl_mask = np.logical_not(
                        np.logical_and(np.logical_not(input_mask), input_wl_mask)
                    )

                    # Determine which spectra to keep
                    # Will mask out any spectrum with at least one masked wavelength within the valid wavelength range
                    mask_spec = np.min(valid_wl_mask, axis=-1, keepdims=True)

                    # Determine which SNe to keep
                    # Will mask out any SN with *no* unmasked spectra
                    mask_sn = np.max(mask_spec, axis=-2)[:, 0]

                    chains = {
                        stage.name: self.results.stages[dt[:-1]][
                            str(stage.stage)
                        ].latents[mask_sn]
                        for stage in self.run_stages
                    }

                    DistributionPlotter.plot_corner(
                        chains,
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud=True,
                        smooth=0,
                        bins=self.sn_dim,
                    )

    @override
    def _clear(
        self: Self,
        *args: "Any",
        setup: bool = False,
        run: bool = False,
        save: bool = False,
        load: bool = False,
        result: bool = False,
        analyse: bool = False,
        **kwargs: "Any",
    ) -> None:
        if setup:
            self.clear_attributes(self.setup_attributes)

        if run:
            self.clear_attributes(self.run_attributes)

        if save:
            self.clear_attributes(self.save_attributes)

        if load:
            self.clear_attributes(self.load_attributes)

        if result:
            self.clear_attributes("results")

        if analyse:
            self.analysis = self.options.analysis or PAEStepAnalysis()

        super()._clear(
            *args,
            setup=setup,
            run=run,
            save=save,
            load=load,
            result=result,
            analyse=analyse,
            **kwargs,
        )

    # === Instance Methods ===

    def setup_data_masks(self: Self) -> None:
        for mask_type in ["train_", "test_", "val_", ""]:
            data: LazySNPAEData = getattr(self, f"{mask_type}data")
            input_redshift = data.redshift
            input_phase = data.phase
            input_wavelength = data.wavelength
            input_mask = data.mask
            data.clear()

            min_redshift: float = getattr(self, f"min_{mask_type}redshift")
            max_redshift: float = getattr(self, f"max_{mask_type}redshift")
            redshift_mask = (
                (input_redshift >= min_redshift) & (input_redshift <= max_redshift)
            )[:, 0:1, 0:1]
            # Mask out SNe outside the redshift range
            sn_mask = redshift_mask.astype(np.int32)

            min_phase: float = getattr(self, f"min_{mask_type}phase")
            max_phase: float = getattr(self, f"max_{mask_type}phase")
            phase_mask = ((input_phase >= min_phase) & (input_phase <= max_phase))[
                ..., 0:1
            ]
            # Mask out spectra outside the phase range
            spec_mask = phase_mask.astype(np.int32)

            min_wavelength: float = getattr(self, f"min_{mask_type}wavelength")
            max_wavelength: float = getattr(self, f"max_{mask_type}wavelength")
            wavelength_mask = (input_wavelength >= min_wavelength) & (
                input_wavelength <= max_wavelength
            )
            # Mask out wavelengths outside the wavelength range
            wl_mask = wavelength_mask.astype(np.int32)

            setattr(self, f"{mask_type}mask", input_mask)
            setattr(self, f"{mask_type}sn_mask", sn_mask)
            setattr(self, f"{mask_type}spec_mask", spec_mask)
            setattr(self, f"{mask_type}wl_mask", wl_mask)


class PAEStep(Model[PAEStepConfig, PAE]):
    id: "ClassVar[str]" = "pae"
    model_backend: "ClassVar[dict[str, Callable[[], type[PAEModel]]]]" = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFPAEModel,
    }

    def __init__(self: Self, config: "PAEStepConfig") -> None:
        super().__init__(config)

        self.bases: dict[str, dict[str, Any]] = {}
        self.plots: dict[str, dict[str, Any]] = {}

    @override
    def _analyse(
        self: Self,
        *args: "Any",
        variants: str | list[str] | None = None,
        **kwargs: "Any",
    ) -> None:
        if variants is None:
            return
        if not isinstance(variants, list):
            variants = [variants]

        for variant_name in variants:
            variant = self.variants[variant_name]

            super()._analyse(*args, **{**kwargs, "variants": [variant_name]})

            labels = {}
            ind = 0
            if variant.physical_latents:
                labels[0] = "ΔAᵥ"
                ind = 1
                labels[variant.n_pae_latents - 2] = "Δℳ"
                labels[variant.n_pae_latents - 1] = "Δp"
            for i in range(variant.n_z_latents):
                labels[ind] = f"z{i + 1}"
                ind += 1

            for dt in ["train_", "test_"]:
                for stage in variant.run_stages:
                    results = variant.results.stages[dt[:-1]][str(stage.stage)]

                    input_mask = results.input_mask
                    input_sn_mask = results.input_sn_mask
                    input_spec_mask = results.input_spec_mask
                    input_wl_mask = results.input_wl_mask

                    if variant.analysis.plot_comparison is not None:
                        for opts in variant.analysis.plot_comparison:
                            o = opts.model_copy(deep=True)
                            o.name = "comparison"
                            self.log.debug(f"Plotting {o.name}")
                            o.plot_kwargs = {
                                "label": f"{variant.name}\n({results.loss:.2E})",
                                "title": f"{self.name} {o.name}",
                            }

                            name = f"{dt[:-1]}/{stage.stage}/{o.name}.{o.ext}"
                            self.plots[name] = self.plots.get(
                                name, {"fig": None, "ax": None, "base": True}
                            )
                            fig = self.plots[name]["fig"]
                            ax = self.plots[name]["ax"]
                            o.plot_base = self.plots[name]["base"]

                            data = getattr(variant, f"{dt}data")
                            (
                                wl,
                                amplitude,
                                sigma,
                                _sn_name,
                                _time,
                                mask,
                                _sn_mask,
                                _spec_mask,
                                _wl_mask,
                            ) = SpectraPlotter.prep(
                                data,
                                o,
                                mask=input_mask,
                                sn_mask=input_sn_mask,
                                spec_mask=input_spec_mask,
                                wl_mask=input_wl_mask,
                            )
                            o.base_wl = wl
                            o.base_amp = amplitude
                            o.base_sigma = sigma
                            o.base_mask = np.logical_not(mask)

                            data.amplitude = results.output_amp
                            data.sigma *= 0

                            fig, ax = SpectraPlotter.plot_comparison(
                                data,
                                o,
                                mask=input_mask,
                                sn_mask=input_sn_mask,
                                spec_mask=input_spec_mask,
                                wl_mask=input_wl_mask,
                                fig=fig,
                                ax=ax,
                                save=False,
                                force=True,
                            )

                            self.plots[name]["fig"] = fig
                            self.plots[name]["ax"] = ax
                            self.plots[name]["base"] = False

    @override
    @callback
    def analyse(self: Self, *args: "Any", **kwargs: "Any") -> None:
        super().analyse(*args, **kwargs)
        if len(self.variants) > 1:
            for name, opts in self.plots.items():
                savepath = self.paths.plots / name
                savepath.parent.mkdir(parents=True, exist_ok=True)
                if savepath.exists():
                    continue
                self.log.debug(f"Plotting {name}")
                fig = opts["fig"]
                ax = opts["ax"]
                fig = Plotter.save(fig, savepath)
                Plotter.close(fig, ax)


PAEStep.register_step(PAE)
