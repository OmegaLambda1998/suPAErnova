# Copyright 2025 Patrick Armstrong
from typing import TYPE_CHECKING, override
from pathlib import Path
import importlib

import numpy as np

from supaernova.steps import Step
from supaernova.analysis import Plotter
from supaernova.steps.models import Model
from supaernova.analysis.spectra import SpectraPlotter
from supaernova.configs.steps.pae import PAEStage, PAEStepResult, PAEStepAnalysis
from supaernova.configs.steps.data import DataStepResult
from supaernova.analysis.distribution import DistributionPlotter

if TYPE_CHECKING:
    from typing import Literal, ClassVar
    from collections.abc import Callable

    from numpy import typing as npt

    from supaernova.steps.data import Data
    from supaernova.configs.steps.pae import PAEStepConfig

    from .tf import TFPAEModel

    PAEModel = TFPAEModel


class PAE(Step):
    def __init__(self, config: "PAEStepConfig") -> None:
        super().__init__(config)

        # === Previous Step Variables ===
        self.data_step: Data
        self.kfold = self.options.kfold

        self.data: DataStepResult
        self.spec_mask: npt.NDArray[bool]
        self.sn_mask: npt.NDArray[bool]

        self.train_data: DataStepResult
        self.train_spec_mask: npt.NDArray[bool]
        self.train_sn_mask: npt.NDArray[bool]

        self.test_data: DataStepResult
        self.test_spec_mask: npt.NDArray[bool]
        self.test_sn_mask: npt.NDArray[bool]

        self.val_data: DataStepResult
        self.val_spec_mask: npt.NDArray[bool]
        self.val_sn_mask: npt.NDArray[bool]

        # === Config Variables ===
        # --- Required ---
        self.physical_latents: bool = self.options.physical_latents
        self.n_physical_latents = 3 if self.physical_latents else 0
        self.seperate_latent_training: bool = self.options.seperate_latent_training
        self.seperate_z_latent_training: bool = self.options.seperate_z_latent_training
        self.validation_frac: float = self.options.validation_frac
        self.batch_size: int

        # --- Optional ---
        self.debug: bool = self.options.debug
        self.profile: bool = self.options.profile
        self.architecture: Literal["dense", "convolutional"]
        self.encode_dims: tuple[int, ...]
        self.decode_dims: tuple[int, ...]
        self.n_z_latents: int = self.options.n_z_latents
        self.n_pae_latents = self.n_physical_latents + self.n_z_latents
        self.batch_normalisation: bool
        self.dropout: float
        self.save_best: bool
        self.patience: float | int
        self.min_phase: float = self.options.min_phase
        self.max_phase: float = self.options.max_phase
        self.min_redshift: float = self.options.min_redshift
        self.max_redshift: float = self.options.max_redshift
        self.min_train_phase: float = self.options.min_train_phase or self.min_phase
        self.max_train_phase: float = self.options.max_train_phase or self.max_phase
        self.min_test_phase: float = self.options.min_test_phase or self.min_phase
        self.max_test_phase: float = self.options.max_test_phase or self.max_phase
        self.min_val_phase: float = self.options.min_val_phase or self.min_phase
        self.max_val_phase: float = self.options.max_val_phase or self.max_phase
        self.min_train_redshift: float = (
            self.options.min_train_redshift or self.min_redshift
        )
        self.max_train_redshift: float = (
            self.options.max_train_redshift or self.max_redshift
        )
        self.min_test_redshift: float = (
            self.options.min_test_redshift or self.min_redshift
        )
        self.max_test_redshift: float = (
            self.options.max_test_redshift or self.max_redshift
        )
        self.min_val_redshift: float = (
            self.options.min_val_redshift or self.min_redshift
        )
        self.max_val_redshift: float = (
            self.options.max_val_redshift or self.max_redshift
        )
        self.phase_offset_scale: float
        self.amplitude_offset_scale: float
        self.mask_fraction: float
        self.loss_residual_penalty: float
        self.loss_delta_av_penalty: float
        self.loss_delta_m_penalty: float
        self.loss_delta_p_penalty: float
        self.loss_covariance_penalty: float
        self.loss_decorrelate_all: bool
        self.loss_decorrelate_dust: bool
        self.loss_clip_delta: float
        self.delta_av_epochs: float
        self.delta_av_lr: float
        self.delta_av_lr_decay_steps: int
        self.delta_av_lr_decay_rate: float
        self.delta_av_lr_weight_decay_rate: float
        self.zs_epochs: int
        self.zs_lr: float
        self.zs_lr_decay_steps: int
        self.zs_lr_decay_rate: float
        self.zs_lr_weight_decay_rate: float
        self.delta_m_epochs: int
        self.delta_m_lr: float
        self.delta_m_lr_decay_steps: int
        self.delta_m_lr_decay_rate: float
        self.delta_m_lr_weight_decay_rate: float
        self.delta_p_epochs: int
        self.delta_p_lr: float
        self.delta_p_lr_decay_steps: int
        self.delta_p_lr_decay_rate: float
        self.delta_p_lr_weight_decay_rate: float
        self.final_epochs: float
        self.final_lr: float
        self.final_lr_decay_steps: int
        self.final_lr_decay_rate: float
        self.final_lr_weight_decay_rate: float

        # === Setup Variables ===
        self.model: PAEModel

        # Data Dimensions
        self.sn_dim: int
        self.spec_dim: int
        self.wl_dim: int
        self.phase_dim: int = 1

        # PAEStages
        self.stage_delta_av: PAEStage
        self.stage_zs: list[PAEStage]
        self.stage_delta_m: PAEStage
        self.stage_delta_p: PAEStage
        self.stage_final: PAEStage
        self.run_stages: list[PAEStage]

        # === Run Variables ===
        # === Result Variables ===
        self.results: PAEStepResult

        # === Analysis Variables ===
        self.analysis: PAEStepAnalysis = self.options.analysis or PAEStepAnalysis()

    @override
    def _setup(self, *, data: "Data") -> None:
        super()._setup()
        # === Previous Step Variables ===
        self.data_step = data
        self.data = data.data
        self.train_data = data.train_data[self.kfold % len(data.train_data)]
        self.test_data = data.test_data[self.kfold % len(data.test_data)]
        self.val_data = self.test_data
        if self.validation_frac > 0:
            ind_split = int(self.data_step.sn_dim * self.validation_frac)
            self.val_data = DataStepResult.model_validate({
                k: v[-ind_split:] for k, v in self.train_data.model_dump().items()
            })
            self.train_data = DataStepResult.model_validate({
                k: v[:-ind_split] for k, v in self.train_data.model_dump().items()
            })

        self.setup_data_masks()

        # === Config Variables ===
        # --- Required ---
        n_batches = self.options.n_batches
        self.batch_size = max(
            int(self.data_step.train_frac * self.data_step.sn_dim / n_batches), 1
        )

        # === Setup Variables ===
        # Data Dimensions
        self.sn_dim = self.data_step.sn_dim
        self.spec_dim = self.data_step.spec_dim
        self.wl_dim = self.data_step.wl_dim

        # PAEStages
        stage_data = {
            "data": self.data,
            "spec_mask": self.spec_mask,
            "sn_mask": self.sn_mask,
            "wl_mask": self.data.wl_mask,
            "train_data": self.train_data,
            "train_spec_mask": self.train_spec_mask,
            "train_sn_mask": self.train_sn_mask,
            "train_wl_mask": self.train_data.wl_mask,
            "test_data": self.test_data,
            "test_spec_mask": self.test_spec_mask,
            "test_sn_mask": self.test_sn_mask,
            "test_wl_mask": self.test_data.wl_mask,
            "val_data": self.val_data,
            "val_spec_mask": self.val_spec_mask,
            "val_sn_mask": self.val_sn_mask,
            "val_wl_mask": self.val_data.wl_mask,
            "debug": self.debug,
            "profile": self.profile,
        }

        self.stage_delta_av = PAEStage.model_validate({
            "stage": 1,
            "prev_stage": None,
            "name": "ΔAᵥ",
            "fname": "delta_av",
            "epochs": self.options.delta_av_epochs,
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

        self.stage_delta_m = PAEStage.model_validate({
            "stage": z0 + self.n_z_latents,
            "prev_stage": z0 + self.n_z_latents - 1,
            "name": "Δℳ",
            "fname": "delta_m",
            "epochs": self.options.delta_m_epochs,
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
    def _completed(self) -> bool:
        final_savepath = self.paths.results / self.model.name / self.model.ckpt_path
        if not (final_savepath.exists() and any(final_savepath.iterdir())):
            self.log.debug(
                f"{self.name} has not completed as {final_savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        final_stage = self.run_stages[-1]
        final_stage.prev_stage = None
        self.model.stage = final_stage
        final_loadpath = self.paths.results / self.model.name

        self.log.debug(f"Loading final PAE model weights from {final_loadpath}")
        self.model.load_checkpoint(final_loadpath, reset_weights=False)

    @override
    def _run(self) -> None:
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

        final_savepath = self.paths.results / self.model.name
        self.log.debug(f"Saving final PAE model weights to {final_savepath}")
        self.model.save_checkpoint(final_savepath)
        self.model = self.model.__class__(self)
        final_stage = self.run_stages[-1]
        final_stage.prev_stage = None
        self.model.stage = final_stage
        final_loadpath = self.paths.results / self.model.name
        self.log.debug(f"Loading final PAE model weights from {final_loadpath}")
        self.model.load_checkpoint(final_loadpath, reset_weights=False)

    @override
    def _result(self) -> None:
        final_stage = self.run_stages[-1]
        final_stage.prev_stage = None
        self.model.stage = final_stage
        final_loadpath = self.paths.results / self.model.name

        self.log.debug(f"Loading final PAE model weights from {final_loadpath}")
        self.model.load_checkpoint(final_loadpath, reset_weights=False)

        self.log.debug("Calculating PAE results")
        dt_results: dict[str, dict[str, PAEStepResult]] = {}
        for dt in ["train", "test"]:
            data = getattr(self, f"{dt}_data")
            model_results: dict[str, PAEStepResult] = {}
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

                input_phase = data.time
                input_amplitude = data.amplitude
                input_d_amplitude = data.sigma
                input_mask = data.mask

                mask = (
                    input_mask
                    * getattr(self.model.stage, f"{dt}_sn_mask")
                    * getattr(self.model.stage, f"{dt}_spec_mask")
                )

                latents, output_amplitude = self.model(
                    (input_phase, input_amplitude),
                    training=False,
                    mask=mask,
                    wl_mask=data.wl_mask,
                )

                loss = self.model.compute_loss(
                    latents,
                    input_amplitude,
                    output_amplitude,
                    sample_weight=input_d_amplitude,
                    training=False,
                    mask=mask,
                )

                pred_loss = self.model.get_loss("loss_pred")
                model_loss = self.model.get_loss("loss_model")
                resid_loss = self.model.get_loss("loss_resid")
                delta_loss = self.model.get_loss("loss_delta")
                cov_loss = self.model.get_loss("loss_cov")

                results = {
                    "stage": stage.stage,
                    "ind": data.ind,
                    "sn_name": data.sn_name,
                    "spectra_id": data.spectra_id,
                    "input_amp": data.amplitude,
                    "input_d_amp": data.sigma,
                    "input_phase": data.time,
                    "input_mask": np.array(mask),
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
                stage_results = PAEStepResult.model_validate(results)
                model_results[str(stage.stage)] = stage_results
            dt_results[dt] = model_results
        self.results = dt_results

    @override
    def _analyse(self) -> None:
        labels = {}
        ind = 0
        if self.physical_latents:
            labels[0] = "ΔAᵥ"
            ind = 1
            labels[self.n_pae_latents - 2] = "Δℳ"
            labels[self.n_pae_latents - 1] = "Δp"
        for i in range(self.n_z_latents):
            labels[ind] = f"z{i}"
            ind += 1

        for dt in ["train", "test"]:
            for stage in self.run_stages:
                if self.analysis.plot_residual is not None:
                    if not isinstance(self.analysis.plot_residual, list):
                        self.analysis.plot_residual = [self.analysis.plot_residual]
                    for opts in self.analysis.plot_residual:
                        o = opts.model_copy()
                        if o.name is None:
                            o.name = "residual"
                        self.log.debug(f"Plotting {o.name}")
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
                                / dt
                                / str(self.model.seed)
                                / str(stage.stage)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        SpectraPlotter.plot_residual(
                            getattr(self, f"{dt}_data"),
                            self.results[dt][str(stage.stage)].output_amp,
                            o,
                        )

                if self.analysis.plot_latents is not None:
                    if not isinstance(self.analysis.plot_latents, list):
                        self.analysis.plot_latents = [self.analysis.plot_latents]
                    for opts in self.analysis.plot_latents:
                        o = opts.model_copy()
                        if o.labels is None:
                            o.labels = {i: labels[i] for i in range(stage.stage)}
                        if o.name is None:
                            o.name = "latents"
                        self.log.debug(f"Plotting {o.name}")
                        if o.savepath is None:
                            o.savepath = (
                                self.paths.plots
                                / dt
                                / str(self.model.seed)
                                / str(stage.stage)
                            )
                        o.savepath.mkdir(parents=True, exist_ok=True)
                        chains = self.results[dt][str(stage.stage)].latents[
                            :, : stage.stage
                        ]
                        DistributionPlotter.plot_corner(
                            chains,
                            o,
                            statistics="max_central",
                            shade_alpha=0.0,
                            plot_cloud=True,
                        )

            if self.analysis.plot_residual is not None:
                if not isinstance(self.analysis.plot_residual, list):
                    self.analysis.plot_residual = [self.analysis.plot_residual]
                for opts in self.analysis.plot_residual:
                    o = opts.model_copy()
                    if o.name is None:
                        o.name = "residual"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)

                    savepath = (o.savepath or Path()) / f"{o.name}.{o.ext}"
                    if not savepath.exists():
                        fig, ax = SpectraPlotter.plot_residual(
                            getattr(self, f"{dt}_data"),
                            self.results[dt][str(self.run_stages[0].stage)].output_amp,
                            o,
                            save=False,
                        )
                        for stage in self.run_stages[1:]:
                            fig, ax = SpectraPlotter.plot_residual(
                                getattr(self, f"{dt}_data"),
                                self.results[dt][str(stage.stage)].output_amp,
                                o,
                                fig=fig,
                                ax=ax,
                                save=False,
                            )

                        fig = Plotter.save(fig, savepath)
                        Plotter.close(fig, ax)

            if self.analysis.plot_latents is not None:
                if not isinstance(self.analysis.plot_latents, list):
                    self.analysis.plot_latents = [self.analysis.plot_latents]
                for opts in self.analysis.plot_latents:
                    o = opts.model_copy()
                    if o.labels is None:
                        o.labels = {
                            stage.name: {i: labels[i] for i in range(stage.stage)}
                            for stage in self.run_stages
                        }
                    if o.name is None:
                        o.name = "latents"
                    self.log.debug(f"Plotting {o.name}")
                    if o.savepath is None:
                        o.savepath = self.paths.plots / dt / str(self.model.seed)
                    o.savepath.mkdir(parents=True, exist_ok=True)

                    chains = {
                        stage.name: self.results[dt][str(stage.stage)].latents
                        for stage in self.run_stages
                    }

                    DistributionPlotter.plot_corner(
                        chains,
                        o,
                        statistics="max_central",
                        shade_alpha=0.0,
                        plot_cloud=True,
                    )

    # === Instance Methods ===

    def setup_data_masks(self) -> None:
        for mask_type in ["train_", "test_", "val_", ""]:
            data: DataStepResult = getattr(self, f"{mask_type}data")
            min_redshift: float = getattr(self, f"min_{mask_type}redshift")
            max_redshift: float = getattr(self, f"max_{mask_type}redshift")
            redshift_mask = (
                (data.redshift >= min_redshift) & (data.redshift <= max_redshift)
            )[:, 0:1, 0:1]

            min_phase: float = getattr(self, f"min_{mask_type}phase")
            max_phase: float = getattr(self, f"max_{mask_type}phase")
            phase_mask = ((data.phase >= min_phase) & (data.phase <= max_phase))[
                ..., 0:1
            ]

            # Mask out spectra outside the phase range
            spec_mask = phase_mask.astype(np.int32)

            # Mask out SNe with no valid spectra
            # Mask out SNe outside the redshift range
            sn_mask = (
                np.any(phase_mask, axis=(1, 2), keepdims=True) & redshift_mask
            ).astype(np.int32)

            setattr(self, f"{mask_type}spec_mask", spec_mask)
            setattr(self, f"{mask_type}sn_mask", sn_mask)


class PAEStep(Model):
    id: "ClassVar[str]" = "pae"
    model_backend: "ClassVar[dict[str, Callable[[], type[PAEModel]]]]" = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFPAEModel,
    }


PAEStep.register_step(PAE)
