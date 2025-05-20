# Copyright 2025 Patrick Armstrong

from typing import TYPE_CHECKING, ClassVar, override
import importlib

import numpy as np

from suPAErnova.steps.backends import AbstractModel
from suPAErnova.configs.steps.pae import PAEStage, PAEStepResult

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from collections.abc import Callable

    from suPAErnova.steps.data import DataStep
    from suPAErnova.configs.paths import PathConfig
    from suPAErnova.configs.globals import GlobalConfig
    from suPAErnova.configs.steps.data import DataStepResult
    from suPAErnova.configs.steps.pae.model import PAEModelConfig

    from .tf import TFPAEModel
    from .tch import TCHPAEModel

    PAEModel = TFPAEModel | TCHPAEModel


class PAEModelStep[Backend: str](AbstractModel[Backend]):
    # --- Class Variables ---
    model_backend: ClassVar[dict[str, "Callable[[], type[PAEModelConfig]]"]] = {
        "TensorFlow": lambda: importlib.import_module(".tf", __package__).TFPAEModel,
        "PyTorch": lambda: importlib.import_module(".tch", __package__).TCHPAEModel,
    }
    id: ClassVar[str] = "pae_model"

    def __init__(self, config: "PAEModelConfig") -> None:
        # --- Superclass Variables ---
        self.options: PAEModelConfig
        self.config: GlobalConfig
        self.paths: PathConfig
        self.log: Logger
        self.force: bool
        self.verbose: bool
        self.model: PAEModel
        super().__init__(config)

        # --- Config Variables ---
        # Required
        self.physical_latents: bool
        self.n_physical: int
        self.n_zs: int
        self.n_pae_latents: int

        self.seperate_latent_training: bool
        self.seperate_z_latent_training: bool

        self.debug: bool

        # Optional
        self.min_train_redshift: float
        self.max_train_redshift: float
        self.min_test_redshift: float
        self.max_test_redshift: float
        self.min_val_redshift: float
        self.max_val_redshift: float

        self.min_train_phase: float
        self.max_train_phase: float
        self.min_test_phase: float
        self.max_test_phase: float
        self.min_val_phase: float
        self.max_val_phase: float

        # --- Previous Step Variables ---
        self.data: DataStep
        self.train_data: DataStepResult
        self.test_data: DataStepResult
        self.val_data: DataStepResult

        self.nspec_dim: int
        self.wl_dim: int
        self.phase_dim: int

        # --- Setup Variables ---
        self.stage_delta_av: PAEStage
        self.stage_zs: list[PAEStage]
        self.stage_delta_m: PAEStage
        self.stage_delta_p: PAEStage
        self.stage_final: PAEStage
        self.run_stages: list[PAEStage]

        self.results: dict[str, PAEStepResult]

    @override
    def _setup(
        self,
        *,
        data: "DataStep",
        train_data: "DataStepResult",
        test_data: "DataStepResult",
        val_data: "DataStepResult",
    ) -> None:
        # --- Config Variables ---
        # Required
        self.physical_latents = self.options.physical_latents
        self.n_physical = 3 if self.options.physical_latents else 0
        self.n_zs = self.options.n_z_latents
        self.n_pae_latents = self.n_physical + self.n_zs

        self.seperate_latent_training = self.options.seperate_latent_training
        self.seperate_z_latent_training = self.options.seperate_z_latent_training

        self.debug = self.options.debug

        # Optional
        self.min_train_redshift = self.options.min_train_redshift
        self.max_train_redshift = self.options.max_train_redshift
        self.min_test_redshift = self.options.min_test_redshift
        self.max_test_redshift = self.options.max_test_redshift
        self.min_val_redshift = self.options.min_val_redshift
        self.max_val_redshift = self.options.max_val_redshift

        self.min_train_phase = self.options.min_train_phase
        self.max_train_phase = self.options.max_train_phase
        self.min_test_phase = self.options.min_test_phase
        self.max_test_phase = self.options.max_test_phase
        self.min_val_phase = self.options.min_val_phase
        self.max_val_phase = self.options.max_val_phase

        # --- Previous Step Variables ---
        self.data = data
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.setup_data_masks()

        self.nspec_dim = self.data.nspec_dim
        self.wl_dim = self.data.wl_dim
        self.phase_dim = 1

        # --- PAEStages ---
        stage_data = {
            "train_data": self.train_data,
            "test_data": self.test_data,
            "val_data": self.val_data,
            "moving_means": [0 for _ in range(self.n_pae_latents)],
            "debug": self.debug,
        }

        self.stage_delta_av = PAEStage.model_validate({
            "stage": 1,
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
                "name": f"z{i + 1}",
                "fname": f"z{i + 1}",
                "epochs": self.options.zs_epochs,
                "learning_rate": self.options.zs_lr,
                "learning_rate_decay_steps": self.options.zs_lr_decay_steps,
                "learning_rate_decay_rate": self.options.zs_lr_decay_rate,
                "learning_rate_weight_decay_rate": self.options.zs_lr_weight_decay_rate,
                **stage_data,
            })
            for i in range(self.n_zs)
        ]
        if not self.seperate_z_latent_training:
            self.stage_zs = [self.stage_zs[-1]]

        self.stage_delta_m = PAEStage.model_validate({
            "stage": z0 + self.n_zs,
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
            "stage": z0 + self.n_zs + 1,
            "name": "Δ𝓅",
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
        self._model()
        final_stage = self.run_stages[-1]
        final_savepath = (
            self.paths.out
            / self.model.name
            / f"{final_stage.stage}_{final_stage.fname}"
            / "checkpoint"
        )

        if not final_savepath.exists():
            self.log.debug(
                f"{self.name} has not completed as {final_savepath} does not exist"
            )
            return False
        return True

    @override
    def _load(self) -> None:
        self._model()

        final_stage = self.run_stages[-1]
        self.model.stage = final_stage
        final_savepath = (
            self.paths.out
            / self.model.name
            / f"{final_stage.stage}_{final_stage.fname}"
        )

        self.log.debug(f"Loading final PAE model weights from {final_savepath}")
        self.model.load_checkpoint(final_savepath, reset_weights=False)

        self._result()

    @override
    def _run(self) -> None:
        savepath: Path | None = None
        for i, stage in enumerate(self.run_stages):
            self._model(force=True)
            self.log.debug(f"Starting PAEStage {i}: {stage.name}")
            if savepath is not None:
                stage.loadpath = savepath
            savepath = self.paths.out / self.model.name / f"{stage.stage}_{stage.fname}"
            stage.savepath = savepath

            model_path = savepath / self.model.model_path
            weights_path = savepath / self.model.weights_path
            if self.force or not model_path.exists():
                self.model.train_model(stage)
                self.model.save_checkpoint(savepath)
            else:
                # Don't retrain stages if you don't need to
                self.log.debug(f"Loading stage weights from {weights_path}")
                self.model.stage = stage
                self.model.load_checkpoint(savepath, reset_weights=False)

    @override
    def _result(self) -> None:
        self._model()
        final_stage = self.run_stages[-1]
        self.model.stage = final_stage
        final_savepath = (
            self.paths.out
            / self.model.name
            / f"{final_stage.stage}_{final_stage.fname}"
        )

        self.log.debug(f"Saving final PAE model weights to {final_savepath}")
        self.model.save_checkpoint(final_savepath)

        self.log.debug("Calculating PAE results")
        data = self.data.data
        model_results: dict[str, PAEStepResult] = {}

        for stage in self.run_stages:
            self._model(force=True)
            savepath = self.paths.out / self.model.name / f"{stage.stage}_{stage.fname}"
            self.model.stage = stage
            self.model.load_checkpoint(savepath, reset_weights=False)
            print(self.model.encoder.moving_means)

            input_phase = data.phase
            input_amplitude = data.amplitude
            input_d_amplitude = data.sigma
            input_mask = data.mask

            latents, output_amplitude = self.model(
                (input_phase, input_amplitude), training=False, mask=input_mask
            )

            loss = self.model.compute_loss(
                latents,
                input_amplitude,
                output_amplitude,
                sample_weight=input_d_amplitude,
                training=False,
                mask=input_mask,
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
                "input_phase": data.phase,
                "input_mask": data.mask,
                "latents": latents.numpy()[:, 0, :],
                "output_amp": output_amplitude.numpy(),
                "diff_amp": abs(input_amplitude - output_amplitude.numpy()),
                "encoder_weights": [w.numpy() for w in self.model.encoder.weights],
                "decoder_weights": [w.numpy() for w in self.model.decoder.weights],
                "loss": loss,
                "pred_loss": pred_loss,
                "model_loss": model_loss,
                "resid_loss": resid_loss,
                "delta_loss": delta_loss,
                "cov_loss": cov_loss,
            }
            stage_results = PAEStepResult.model_validate(results)
            model_results[str(stage.stage)] = stage_results
        self.results = model_results

    @override
    def _analyse(self) -> None:
        pass

    #
    # === PAEModel Specific Functions ===
    #

    def setup_data_masks(self) -> None:
        for mask_type in ["train", "test", "val"]:
            data: DataStepResult = getattr(self, f"{mask_type}_data")
            min_redshift: float = getattr(self, f"min_{mask_type}_redshift")
            max_redshift: float = getattr(self, f"max_{mask_type}_redshift")
            redshift_mask = (data.redshift >= min_redshift) & (
                data.redshift <= max_redshift
            )

            min_phase: float = getattr(self, f"min_{mask_type}_phase")
            max_phase: float = getattr(self, f"max_{mask_type}_phase")
            phase_mask = (data.phase >= min_phase) & (data.phase <= max_phase)

            mask = (redshift_mask & phase_mask).astype(np.int32)
            data.mask *= mask
            setattr(self, f"{mask_type}_data", data)
