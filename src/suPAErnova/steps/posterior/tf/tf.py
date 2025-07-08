# Copyright 2025 Patrick Armstrong
import os
import random as rn
from typing import TYPE_CHECKING, cast, override

from tqdm import tqdm
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

if TYPE_CHECKING:
    from typing import Any, Self, Literal
    from logging import Logger
    from pathlib import Path
    from collections.abc import Sequence

    from numpy import typing as npt

    from suPAErnova.steps.pae.tf import S, TFPAEModel, TensorCompatible
    from suPAErnova.steps.nflow.tf import TFNFlowModel
    from suPAErnova.configs.steps.data import DataStepResult
    from suPAErnova.steps.posterior.model import PosteriorModelStep
    from suPAErnova.configs.steps.posterior.tf import TFPosteriorModelConfig
    from suPAErnova.configs.steps.posterior.posterior import PosteriorMapStage


class PosteriorMapValue(tf.Module):
    def __init__(self, initial: tf.Variable) -> None:
        self.original: tf.Variable = initial
        self.initial: tf.Variable = initial
        self.current: tf.Variable = initial
        self.best: tf.Variable = initial


class PosteriorHMCValue(tf.Module):
    def __init__(
        self,
        samples: tf.Variable,
        step_sizes_final: tf.Variable,
        is_accepted: tf.Variable,
        u_delta_av: tf.Variable,
        u_latents: tf.Variable,
        delta_av: tf.Variable,
        z_latents: tf.Variable,
        delta_m: tf.Variable,
        delta_p: tf.Variable,
    ) -> None:
        self.samples: tf.Variable = samples
        self.step_sizes_final: tf.Variable = step_sizes_final
        self.is_accepted: tf.Variable = is_accepted

        self.u_delta_av: tf.Variable = u_delta_av
        self.u_latents: tf.Variable = u_latents
        self.delta_av: tf.Variable = delta_av
        self.z_latents: tf.Variable = z_latents
        self.delta_m: tf.Variable = delta_m
        self.delta_p: tf.Variable = delta_p


class PosteriorMap(tf.Module):
    def __init__(
        self,
        config: "PosteriorModelStep[TFPosteriorModelConfig]",
        nflow: "TFNFlowModel",
        pae: "TFPAEModel",
        data: "DataStepResult",
    ) -> None:
        self.random_initial_positions: bool = config.random_initial_positions
        # Equivalent to `self.name = ...` but avoids tf / ks from tracking self.name
        vars(self)["nflow"]: TFNFlowModel = nflow
        vars(self)["pae"]: TFPAEModel = pae
        self.data: DataStepResult = data

        self.sn_dim = self.data.amplitude.shape[0]
        self.spec_dim = self.data.amplitude.shape[1]
        self.wl_dim = self.data.amplitude.shape[2]
        self.n_u_latents = self.nflow.n_u_latents
        self.n_flow_latents = self.nflow.n_flow_latents
        self.n_z_latents = self.pae.n_z_latents
        self.n_pae_latents = self.pae.n_pae_latents
        self.n_pos = self.n_u_latents

        # === Training ===
        self.chain_min = tf.Variable(tf.zeros(self.sn_dim, dtype=tf.int32))
        self.converged = tf.Variable(
            tf.cast(tf.zeros(self.sn_dim, dtype=tf.int32), tf.bool)
        )
        self.improved = tf.Variable(
            tf.cast(tf.zeros(self.sn_dim, dtype=tf.int32), tf.bool)
        )
        self.num_evaluations = tf.Variable(0, dtype=tf.int32)
        self.num_chain_evaluations = tf.Variable(0, dtype=tf.int32)
        self.negative_log_prob = tf.Variable(
            np.inf * tf.ones(self.sn_dim, dtype=tf.float32)
        )

        # === Priors ===
        self.u_delta_av_min: float = config.u_delta_av_min
        self.u_delta_av_max: float = config.u_delta_av_max
        self.u_delta_av_mean: float = config.u_delta_av_mean
        self.u_delta_av_std: float = config.u_delta_av_std
        self.u_delta_av_prior = tfd.Normal(
            loc=self.u_delta_av_mean, scale=self.u_delta_av_std
        )
        if self.nflow.physical_latents:
            self.n_pos += 1

        self.u_latents_mean: float = config.u_latents_mean
        self.u_latents_std: float = config.u_latents_std
        self.u_latents_prior = tfd.MultivariateNormalDiag(
            loc=self.u_latents_mean * tf.ones(self.n_u_latents),
            scale_diag=self.u_latents_std * tf.ones(self.n_u_latents),
        )

        self.delta_av_min: float = config.delta_av_min
        self.delta_av_max: float = config.delta_av_max
        self.delta_av_mean: float = config.delta_av_mean
        self.delta_av_std: float = config.delta_av_std
        self.delta_av_prior = tfd.Normal(
            loc=self.delta_av_mean, scale=self.delta_av_std
        )

        self.train_delta_m: bool = config.train_delta_m
        self.delta_m_min: float = config.delta_m_min
        self.delta_m_max: float = config.delta_m_max
        self.delta_m_mean: float = config.delta_m_mean
        self.delta_m_std: float = config.delta_m_std
        self.delta_m_prior = tfd.Normal(loc=self.delta_m_mean, scale=self.delta_m_std)
        if self.train_delta_m:
            self.n_pos += 1

        self.train_delta_p: bool = config.train_delta_p
        self.delta_p_min: float = config.delta_p_min
        self.delta_p_max: float = config.delta_p_max
        self.delta_p_mean: float = config.delta_p_mean
        self.delta_p_std: float = config.delta_p_std
        self.delta_p_prior = tfd.Normal(loc=self.delta_p_mean, scale=self.delta_p_std)
        if self.train_delta_p:
            self.n_pos += 1

        self.train_bias: bool = config.train_bias
        self.bias_min: float = config.bias_min
        self.bias_max: float = config.bias_max
        self.bias_mean: float = config.bias_mean
        self.bias_std: float = config.bias_std
        self.bias_prior = tfd.Normal(loc=self.bias_mean, scale=self.bias_std)
        if self.train_bias:
            self.n_pos += 1

        self.u_delta_av: PosteriorMapValue = PosteriorMapValue(
            tf.Variable(np.inf * tf.ones((self.sn_dim, 1)))
        )
        self.u_latents: PosteriorMapValue = PosteriorMapValue(
            tf.Variable(np.inf * tf.ones((self.sn_dim, self.n_u_latents)))
        )
        self.z_latents: PosteriorMapValue = PosteriorMapValue(
            tf.Variable(np.inf * tf.ones((self.sn_dim, self.n_z_latents)))
        )

        self.delta_av: PosteriorMapValue = PosteriorMapValue(
            tf.Variable(np.inf * tf.ones((self.sn_dim, 1)))
        )
        self.delta_m: PosteriorMapValue = PosteriorMapValue(
            tf.Variable(np.inf * tf.ones((self.sn_dim, 1)))
        )
        self.delta_p: PosteriorMapValue = PosteriorMapValue(
            tf.Variable(np.inf * tf.ones((self.sn_dim, 1)))
        )

        self.bias: PosteriorMapValue = PosteriorMapValue(
            tf.Variable(np.inf * tf.ones((self.sn_dim, 1)))
        )
        self.position: PosteriorMapValue = PosteriorMapValue(
            tf.Variable(np.inf * tf.ones((self.sn_dim, self.n_pos)))
        )

        self.labels: list[str] = []
        if self.train_delta_m:
            self.labels.append("Δℳ")
        if self.train_delta_p:
            self.labels.append("Δp")
        if self.train_bias:
            self.labels.append("Bias")
        if self.nflow.physical_latents:
            self.labels.append("μΔAᵥ")
        for i in range(self.n_u_latents):
            self.labels.append(f"μ{i}")

    def setup(
        self,
        stage: "PosteriorMapStage",
        chain: int,
    ) -> None:
        # === Initial Values ===
        # Generate values for all params which will serve as their initial value.
        if stage.init:
            init_all = "random" if self.random_initial_positions else "data"
            stage.init_u_delta_av = init_all
            stage.init_latents = "u_random" if init_all == "random" else "z_data"
            stage.init_delta_av = init_all
            stage.init_delta_m = init_all
            stage.init_delta_p = init_all
            stage.init_bias = init_all
        else:
            # After initialisation:
            # If we're not training a variable, don't bother generating it
            if not self.train_delta_m:
                stage.init_delta_m = "current"
            if not self.train_delta_p:
                stage.init_delta_p = "current"
            if not self.train_bias:
                stage.init_bias = "current"

            # If we're not using u_delta_av in the nflow model, don't generate it
            if not self.nflow.physical_latents:
                stage.init_u_delta_av = "current"

            if not self.pae.physical_latents:
                # If we're not using delta_av in the pae model, don't generate it
                stage.init_delta_av = "current"
                # If we're not using delta_m or delta_p in the pae model, they can't be generated from data, so set to a constant instead
                if stage.init_delta_m == "data":
                    stage.init_delta_m = "constant"
                if stage.init_delta_p == "data":
                    stage.init_delta_p = "constant"
        if stage.init_bias == "data":
            stage.init_bias = "constant"

        if stage.init_latents[0] == "u":
            # If we're generating u_latents, then u_delta_av can't be generated from data, so set it to the same generation as init_latents
            if stage.init_u_delta_av == "data":
                stage.init_u_delta_av = (
                    "random" if stage.init_latents == "u_random" else "constant"
                )
            # If we're generating u_latents, then delta_m and delta_p can't be generate from data, so set to constant instead
            if stage.init_delta_m == "data":
                stage.init_delta_m = "constant"
            if stage.init_delta_p == "data":
                stage.init_delta_p = "constant"

        # === Generating Latent Values ===

        # We are generating u_latents then transforming them to z_latents
        if stage.init_latents[0] == "u":
            if stage.init_latents == "u_random":
                u_latents = self.u_latents_prior.sample(self.sn_dim)
            elif stage.init_latents == "u_constant":
                u_latents = self.u_latents_mean * tf.ones((
                    self.sn_dim,
                    self.n_u_latents,
                ))
            # We need to generate u_delta_av
            if self.nflow.physical_latents:
                if stage.init_u_delta_av == "current":
                    u_delta_av = self.u_delta_av.current
                elif stage.init_u_delta_av == "best":
                    u_delta_av = self.u_delta_av.best
                elif stage.init_u_delta_av == "random":
                    u_delta_av = self.u_delta_av_prior.sample((self.sn_dim, 1))
                elif stage.init_u_delta_av == "constant":
                    u_delta_av = self.u_delta_av_mean * tf.ones((self.sn_dim, 1))
                elif stage.init_u_delta_av == "scale":
                    u_delta_av_slope = (
                        self.u_delta_av_max - self.u_delta_av_min
                    ) / stage.n_chains
                    u_delta_av_scale = (
                        self.u_delta_av_min
                        + (stage.n_chains - chain) * u_delta_av_slope
                    )
                    u_delta_av = tf.ones((self.sn_dim, 1)) * u_delta_av_scale
                us = tf.concat([u_delta_av, u_latents], axis=-1)
            else:
                us = u_latents
            # Transform u_latents to z_latents
            z_latents = self.nflow.u_to_z(us, permute=True)
            if self.nflow.physical_latents:
                if stage.init_delta_av == "data":
                    delta_av = z_latents[:, 0:1]
                z_latents = z_latents[:, 1:]
        # We are generating z_latents then transforming them to u_latents
        elif stage.init_latents[0] == "z":
            if stage.init_latents == "z_data":
                # Generate z_latents directly from data
                pae_input = tf.concat((self.data.time, self.data.amplitude), axis=-1)
                z_latents = self.pae.encoder(
                    pae_input,
                    training=False,
                    mask=self.data.mask,
                )[:, 0, :]
                if self.pae.physical_latents:
                    if stage.init_delta_av == "data":
                        delta_av = z_latents[:, 0:1]
                    if stage.init_delta_m == "data":
                        delta_m = z_latents[
                            :, self.n_z_latents + 1 : self.n_z_latents + 2
                        ]
                    if stage.init_delta_p == "data":
                        delta_p = z_latents[
                            :, self.n_z_latents + 2 : self.n_z_latents + 3
                        ]
                    z_latents = z_latents[:, 1 : self.n_z_latents + 1]
                if self.nflow.physical_latents:
                    zs = tf.concat([delta_av, z_latents], axis=-1)
                else:
                    zs = z_latents
            else:
                # First generate u_latents, then transform to z_latents, finally modify the result somehow.
                if stage.init_latents == "z_random":
                    u_latents = self.u_latents_prior.sample(self.sn_dim)
                elif stage.init_latents == "z_constant":
                    u_latents = self.u_latents_mean * tf.ones((
                        self.sn_dim,
                        self.n_u_latents,
                    ))
                # We need to generate u_delta_av
                if self.nflow.physical_latents:
                    if stage.init_u_delta_av == "current":
                        u_delta_av = self.u_delta_av.current
                    elif stage.init_u_delta_av == "best":
                        u_delta_av = self.u_delta_av.best
                    elif stage.init_u_delta_av == "scale":
                        u_delta_av_slope = (
                            self.u_delta_av_max - self.u_delta_av_min
                        ) / stage.n_chains
                        u_delta_av_scale = (
                            self.u_delta_av_min
                            + (stage.n_chains - chain) * u_delta_av_slope
                        )
                        u_delta_av = tf.ones((self.sn_dim, 1)) * u_delta_av_scale
                    elif stage.init_u_delta_av == "random" or (
                        stage.init_u_delta_av == "data"
                        and stage.init_latents == "z_random"
                    ):
                        u_delta_av = self.u_delta_av_prior.sample((self.sn_dim, 1))
                    elif stage.init_u_delta_av == "constant" or (
                        stage.init_u_delta_av == "data"
                        and stage.init_latents == "z_constant"
                    ):
                        u_delta_av = self.u_delta_av_mean * tf.ones((self.sn_dim, 1))
                    us = tf.concat([u_delta_av, u_latents], axis=-1)
                else:
                    us = u_latents
                # Transform u_latents to z_latents
                zs = self.nflow.u_to_z(us, permute=True)
                if self.nflow.physical_latents:
                    # We want to modify zs
                    if stage.init_delta_av == "best":
                        delta_av = self.delta_av.best
                    elif stage.init_delta_av == "scale":
                        delta_av_slope = (
                            self.delta_av_max - self.delta_av_min
                        ) / stage.n_chains
                        delta_av_scale = (
                            self.delta_av_min
                            + (stage.n_chains - chain) * delta_av_slope
                        )
                        delta_av = tf.ones((self.sn_dim, 1)) * delta_av_scale
                    elif stage.init_delta_av == "random":
                        delta_av = self.delta_av_prior.sample((self.sn_dim, 1))
                    elif stage.init_delta_av == "constant":
                        delta_av = self.delta_av_mean * tf.ones((self.sn_dim, 1))
                    elif stage.init_delta_av == "data":
                        delta_av = zs[:, 0:1]
                    z_latents = zs[:, 1:]
                    zs = tf.concat([delta_av, z_latents], axis=-1)
                else:
                    z_latents = zs

            # After generating z_latents, transform them to u_latents
            u_latents = self.nflow.z_to_u(zs, permute=True)
            if self.nflow.physical_latents:
                if stage.init_u_delta_av == "data":
                    u_delta_av = u_latents[:, 0:1]
                u_latents = u_latents[:, 1:]

        # === Preset Values ===
        # --- Current ---
        if stage.init_u_delta_av == "current":
            u_delta_av = self.u_delta_av.current
        if stage.init_latents == "current":
            u_latents = self.u_latents.current
            z_latents = self.u_latents.current
        if stage.init_delta_av == "current":
            delta_av = self.delta_av.current
        if stage.init_delta_m == "current":
            delta_m = self.delta_m.current
        if stage.init_delta_p == "current":
            delta_p = self.delta_p.current
        if stage.init_bias == "current":
            bias = self.bias.current

        # --- Best ---
        if stage.init_u_delta_av == "best":
            u_delta_av = self.u_delta_av.best
        if stage.init_latents == "best":
            u_latents = self.u_latents.best
            z_latents = self.u_latents.best
        if stage.init_delta_av == "best":
            delta_av = self.delta_av.best
        if stage.init_delta_m == "best":
            delta_m = self.delta_m.best
        if stage.init_delta_p == "best":
            delta_p = self.delta_p.best
        if stage.init_bias == "best":
            bias = self.bias.best

        # At this point, we are certain to have generated u_latents, z_latents, u_delta_av as well as any parameters with "data" generation
        # Now we cover all the other options

        # --- delta_av ---
        # For delta_av to have been generated, it must have occured in either z_random or z_constant
        if not (
            stage.init_latents[0] == "z"
            and stage.init_latents != "z_data"
            and self.nflow.physical_latents
        ):
            if stage.init_delta_av == "scale":
                delta_av_slope = (
                    self.delta_av_max - self.delta_av_min
                ) / stage.n_chains
                delta_av_scale = (
                    self.delta_av_min + (stage.n_chains - chain) * delta_av_slope
                )
                delta_av = tf.ones((self.sn_dim, 1)) * delta_av_scale
            elif stage.init_delta_av == "random":
                delta_av = self.delta_av_prior.sample((self.sn_dim, 1))
            elif stage.init_delta_av == "constant":
                delta_av = self.delta_av_mean * tf.ones((self.sn_dim, 1))

        # --- delta_m ---
        if stage.init_delta_m == "random":
            delta_m = self.delta_m_prior.sample((self.sn_dim, 1))
        elif stage.init_delta_m == "scale":
            delta_m_slope = (self.delta_m_max - self.delta_m_min) / stage.n_chains
            delta_m_scale = self.delta_m_min + (stage.n_chains - chain) * delta_m_slope
            delta_m = tf.zeros((self.sn_dim, 1)) + delta_m_scale
        elif stage.init_delta_m == "constant":
            delta_m = self.delta_m_mean * tf.ones((self.sn_dim, 1))

        # --- delta_p ---
        if stage.init_delta_p == "random":
            delta_p = self.delta_p_prior.sample((self.sn_dim, 1))
        elif stage.init_delta_p == "scale":
            delta_p_slope = (self.delta_p_max - self.delta_p_min) / stage.n_chains
            delta_p_scale = self.delta_p_min + (stage.n_chains - chain) * delta_p_slope
            delta_p = tf.zeros((self.sn_dim, 1)) + delta_p_scale
        elif stage.init_delta_p == "constant":
            delta_p = self.delta_p_mean * tf.ones((self.sn_dim, 1))

        # --- bias ---
        if stage.init_bias == "random":
            bias = self.bias_prior.sample((self.sn_dim, 1))
        elif stage.init_bias in {"scale", "constant"}:
            bias = self.bias_mean * tf.ones((self.sn_dim, 1))

        position = []
        if self.train_delta_m:
            position.append(delta_m)
        if self.train_delta_p:
            position.append(delta_p)
        if self.train_bias:
            position.append(bias)
        if self.nflow.physical_latents:
            position.append(u_delta_av)
        position.append(u_latents)
        position = tf.concat(position, axis=-1)

        self.u_delta_av.current = tf.Variable(u_delta_av)
        self.u_latents.current = tf.Variable(u_latents)
        self.z_latents.current = tf.Variable(z_latents)
        self.delta_av.current = tf.Variable(delta_av)
        self.delta_m.current = tf.Variable(delta_m)
        self.delta_p.current = tf.Variable(delta_p)
        self.bias.current = tf.Variable(bias)
        self.position.current = tf.Variable(position)

        if stage.init:
            self.u_delta_av.original = self.u_delta_av.current
            self.u_delta_av.initial = self.u_delta_av.current
            self.u_delta_av.best = self.u_delta_av.current

            self.u_latents.original = self.u_latents.current
            self.u_latents.initial = self.u_latents.current
            self.u_latents.best = self.u_latents.current

            self.z_latents.original = self.z_latents.current
            self.z_latents.initial = self.z_latents.current
            self.z_latents.best = self.z_latents.current

            self.delta_av.original = self.delta_av.current
            self.delta_av.initial = self.delta_av.current
            self.delta_av.best = self.delta_av.current

            self.delta_m.original = self.delta_m.current
            self.delta_m.initial = self.delta_m.current
            self.delta_m.best = self.delta_m.current

            self.delta_p.original = self.delta_p.current
            self.delta_p.initial = self.delta_p.current
            self.delta_p.best = self.delta_p.current

            self.bias.original = self.bias.current
            self.bias.initial = self.bias.current
            self.bias.best = self.bias.current

            self.position.original = self.position.current
            self.position.initial = self.position.current
            self.position.best = self.position.current

    def get_position(self, position, best=False) -> tf.Tensor:
        u_delta_av = self.u_delta_av.best if best else self.u_delta_av.current
        delta_m = self.delta_m.best if best else self.delta_m.current
        delta_p = self.delta_p.best if best else self.delta_p.current
        bias = self.bias.best if best else self.bias.current

        i = 0
        if self.train_delta_m:
            delta_m = position[:, i : i + 1]
            i += 1
        if self.train_delta_p:
            delta_p = position[:, i : i + 1]
            i += 1
        if self.train_bias:
            bias = position[:, i : i + 1]
            i += 1
        if self.nflow.physical_latents:
            u_delta_av = position[:, i : i + 1]
            i += 1
        u_latents = position[:, i:]

        return tf.concat([delta_m, delta_p, bias, u_delta_av, u_latents], axis=-1)

    def prior(self, position) -> None:
        delta_m = position[:, 0:1]
        delta_p = position[:, 1:2]
        bias = position[:, 2:3]
        u_delta_av = position[:, 3:4]
        u_latents = position[:, 4:]

        log_prior = self.u_latents_prior.log_prob(u_latents)
        # if self.train_delta_m:
        #     log_prior += self.delta_m_prior.log_prob(delta_m)[:, 0]
        # if self.train_delta_p:
        #     log_prior += self.delta_p_prior.log_prob(delta_p)[:, 0]
        # if self.train_bias:
        #     log_prior += self.bias_prior.log_prob(bias)[:, 0]
        if self.nflow.physical_latents:
            log_prior += self.u_delta_av_prior.log_prob(u_delta_av)[:, 0]

        return log_prior


@ks.utils.register_keras_serializable("SuPAErnova")
class TFPosteriorModel(ks.Model):
    def __init__(
        self: "TFPosteriorModel",
        config: "PosteriorModelStep[TFPosteriorModelConfig]",
        *args: "Any",
        **kwargs: "Any",
    ) -> None:
        ks.backend.clear_session()

        super().__init__(
            *args, name=f"{config.name.split()[-1]}PosteriorModel", **kwargs
        )
        # --- Config ---
        global POSTERIORMODELSTEP
        POSTERIORMODELSTEP = config
        options = config.options
        self.log: Logger = config.log
        self.verbose: bool = config.config.verbose
        self.force: bool = config.config.force
        self.seed: int = config.options.seed
        self.subset: Literal["train", "test"] = config.options.subset

        self.debug: bool = options.debug
        self.profile: bool = options.profile
        # Equivalent to `self.name = ...` but avoids tf / ks from tracking self.name
        vars(self)["nflow"]: TFNFlowModel = config.nflow.model
        vars(self)["pae"]: TFPAEModel = self.nflow.pae
        self.nflow.trainable = False
        self.nflow.flow.trainable = False

        self.data: DataStepResult = (
            self.pae.stage.train_data
            if self.subset == "train"
            else self.pae.stage.test_data
        )
        self.sn_mask: npt.NDArray[np.int32] = (
            self.pae.stage.train_sn_mask.astype(np.int32)
            if self.subset == "train"
            else self.pae.stage.test_sn_mask.astype(np.int32)
        )
        self.spec_mask: npt.NDArray[np.int32] = (
            self.pae.stage.train_spec_mask.astype(np.int32)
            if self.subset == "train"
            else self.pae.stage.test_spec_mask.astype(np.int32)
        )
        self.data.mask *= self.spec_mask

        self.sn_dim = self.data.time.shape[0]
        self.spec_dim = self.data.time.shape[1]

        # MAP Variables
        vars(self)["map"]: PosteriorMap = PosteriorMap(
            options, self.nflow, self.pae, self.data
        )
        self.tolerance = options.tolerance
        self.max_iterations = options.max_iterations

        # --- Training ---
        self.batch_size: int = options.batch_size
        self.save_best: bool = options.save_best
        self.ckpt_path: str = (
            f"{'best' if self.save_best else 'latest'}.model.checkpoint/"
        )

        inds = np.squeeze(self.sn_mask.astype(np.bool_), axis=(1, 2))
        self.recon_error, _recon_error_edges, self.recon_error_centers = (
            self.pae.recon_error((
                tf.convert_to_tensor(self.data.time[inds], dtype=tf.float32),
                tf.convert_to_tensor(self.data.amplitude[inds], dtype=tf.float32),
                tf.convert_to_tensor(self.data.sigma[inds], dtype=tf.float32),
                tf.convert_to_tensor(self.data.mask[inds], dtype=tf.int32),
            ))
        )

        loss: ks.losses.Loss = options.loss_cls()
        loss.model = self
        self._loss: ks.losses.Loss = loss

        # HMC Variables
        self.n_burnin: int = options.n_burnin
        self.n_samples: int = options.n_samples
        self.n_leapfrog: int = options.n_leapfrog
        self.target_acceptance_rate: float = options.target_acceptance_rate

        vars(self)["hmc"]: PosteriorHMCValue = PosteriorHMCValue(
            tf.Variable(
                [[[0] * self.map.n_pae_latents] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, self.map.n_pae_latents),
            ),
            tf.Variable(
                [[[0] * self.map.n_pae_latents] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, self.map.n_pae_latents),
            ),
            tf.Variable(
                [[False] * self.sn_dim] * self.n_samples,
                dtype=tf.bool,
                shape=(
                    self.n_samples,
                    self.sn_dim,
                ),
            ),
            tf.Variable(
                [[[0] * 1] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, 1),
            ),
            tf.Variable(
                [[[0] * self.map.n_u_latents] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, self.map.n_u_latents),
            ),
            tf.Variable(
                [[[0] * 1] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, 1),
            ),
            tf.Variable(
                [[[0] * self.map.n_z_latents] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, self.map.n_z_latents),
            ),
            tf.Variable(
                [[[0] * 1] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, 1),
            ),
            tf.Variable(
                [[[0] * 1] * self.sn_dim] * self.n_samples,
                dtype=tf.float32,
                shape=(self.n_samples, self.sn_dim, 1),
            ),
        )

        self.set_seed()

    @override
    def call(
        self,
        inputs: "PosteriorInputs",
        training: bool | None = None,
        mask: "TensorCompatible | None" = None,
    ) -> "PosteriorOutputs":
        training = False if training is None else training

        input_position = inputs[0]
        input_phase = inputs[1]
        input_amp = inputs[2]
        input_sigma = inputs[3]
        input_mask = (
            mask if mask is not None else tf.ones_like(input_amp, dtype=tf.int32)
        )

        delta_m = input_position[:, 0:1]
        delta_p = input_position[:, 1:2]
        bias = input_position[:, 2:3]
        u_delta_av = input_position[:, 3:4]
        u_latents = input_position[:, 4:]

        # Transform from u-latent to z-latent space
        if self.nflow.physical_latents:
            us = tf.concat([u_delta_av, u_latents], axis=-1)
        else:
            us = u_latents

        zs = self.nflow.u_to_z(us, permute=True)
        if self.nflow.physical_latents:
            delta_av = zs[:, 0:1]
            zs = zs[:, 1:]
        if self.pae.physical_latents:
            zs = tf.concat(
                [
                    delta_av,
                    zs,
                    delta_m,
                    delta_p,
                ],
                axis=-1,
            )

        zs = ks.layers.RepeatVector(self.spec_dim)(zs)

        # Create synthetic spectra from z-latents
        decoder_inputs = tf.concat((input_phase, zs), axis=-1)
        synth_amp = self.pae.decoder(decoder_inputs, mask=input_mask)

        # XXX: Test whether this fixes things
        if self.map.train_delta_m:  # and not self.pae.physical_latents:
            delta_m = ks.layers.RepeatVector(1)(delta_m)
            synth_amp *= delta_m

        if self.map.train_bias:
            bias = ks.layers.RepeatVector(1)(bias)
            synth_amp += bias

        delta_p = ks.layers.RepeatVector(1)(delta_p)
        phase = input_phase + delta_p

        # Measured average AE reconstruction error at current times
        sigma_recon = tf.transpose(
            tfp.math.interp_regular_1d_grid(
                x=tf.transpose(phase[:, :, 0]),
                x_ref_min=self.recon_error_centers[0],
                x_ref_max=self.recon_error_centers[-1],
                y_ref=self.recon_error,
            )
        )

        synth_sigma = tf.sqrt(((synth_amp * sigma_recon) ** 2) + (input_sigma**2))

        # Set missing values to 1 for all times
        synth_sigma *= input_mask
        synth_sigma += 1 - input_mask

        # Create likelihood distribution
        likelihood = tfd.Independent(
            tfd.MultivariateNormalDiag(
                loc=synth_amp * tf.cast(input_mask, tf.float32), scale_diag=synth_sigma
            ),
            reinterpreted_batch_ndims=0,
        )

        # Determine the log likelihood
        is_kept = tf.cast(tf.reduce_min(input_mask, axis=-1), tf.float32)
        log_likelihood_num = tf.reduce_sum(
            likelihood.log_prob(input_amp * tf.cast(input_mask, tf.float32)) * is_kept,
            axis=1,
        )
        log_likelihood_sum = tf.reduce_sum(
            tf.cast(tf.math.greater(input_amp[:, :, 0], -1), tf.float32), axis=1
        )
        log_likelihood = log_likelihood_num / log_likelihood_sum

        # Determine the prior probability
        log_prior = self.map.prior(input_position)

        return log_prior + log_likelihood

    def save_checkpoint(
        self, savepath: "Path", *, save_map: bool = False, save_hmc: bool = False
    ) -> None:
        (savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)

        if save_map and save_hmc:
            ckpt = tf.train.Checkpoint(self, map=self.map, hmc=self.hmc)
        elif save_map:
            ckpt = tf.train.Checkpoint(self, map=self.map)
        elif save_hmc:
            ckpt = tf.train.Checkpoint(self, hmc=self.hmc)
        else:
            ckpt = tf.train.Checkpoint(self)

        ckpt.save(f"{savepath / self.ckpt_path}/")

    def load_checkpoint(
        self, loadpath: "Path", *, load_map: bool = False, load_hmc: bool = False
    ) -> None:
        if load_map and load_hmc:
            ckpt = tf.train.Checkpoint(self, map=self.map, hmc=self.hmc)
        elif load_map:
            ckpt = tf.train.Checkpoint(self, map=self.map)
        elif load_hmc:
            ckpt = tf.train.Checkpoint(self, hmc=self.hmc)
        else:
            ckpt = tf.train.Checkpoint(self)

        ckpt.restore(
            tf.train.latest_checkpoint(f"{loadpath / self.ckpt_path}/")
        ).assert_existing_objects_matched()

    @override
    def get_config(self) -> dict[str, "Any"]:
        return {**super().get_config()}

    @override
    @classmethod
    def from_config(cls, config: dict[str, "Any"]) -> "Self":
        global POSTERIORMODELSTEP
        return cls(POSTERIORMODELSTEP)

    @override
    def set_seed(self, seed: int = 0) -> None:
        seed = self.seed + seed
        tf.random.set_seed(seed)

    def train_model(
        self,
        stages: "Sequence[PosteriorMapStage]",
        *,
        savepath: "Path | None" = None,
    ) -> None:
        if savepath is not None and (savepath / self.ckpt_path).exists():
            self.log.debug(f"Loading Posterior from {savepath}")
            self.load_checkpoint(savepath, load_map=True, load_hmc=True)

            return

        n_total = sum(stage.n_chains for stage in stages)
        chain = 0
        progress = tqdm(total=n_total)

        for stage in stages:
            for c in range(stage.n_chains):
                self.set_seed(chain)
                progress.set_description(f"Stage: {chain}/{n_total}")
                progress.set_postfix({
                    "evals_tot": self.map.num_evaluations.value().numpy(),
                    "evals_prev": self.map.num_chain_evaluations.value().numpy(),
                    "improved": tf.reduce_sum(
                        tf.ones_like(self.map.improved, dtype=tf.int32)
                        * tf.cast(self.map.improved, tf.int32)
                    ).numpy(),
                    "neg_log_prob": (
                        f"{float(tf.reduce_min(self.map.negative_log_prob).numpy()):.2E}",
                        f"{float(tf.reduce_mean(self.map.negative_log_prob).numpy()):.2E}",
                        f"{float(tf.reduce_max(self.map.negative_log_prob).numpy()):.2E}",
                    ),
                })
                self.train_map(stage, c, chain, savepath=savepath)
                progress.update()
                chain += 1
        self.log.info(f"Minimum found at chains:\n{self.map.chain_min}")
        self.set_seed()
        self.hmc_train(savepath=savepath)

        if savepath is not None:
            self.save_checkpoint(savepath, save_map=True, save_hmc=True)

    def vals_and_grads(self, position):
        input_position = self.map.get_position(position)
        log_prob = self(
            (input_position, self.data.time, self.data.amplitude, self.data.sigma),
            training=False,
            mask=self.data.mask,
        )
        return self._loss(tf.zeros_like(log_prob), log_prob)

    def lbfgs(self, x):
        return tfp.math.value_and_gradient(
            self.vals_and_grads, x, auto_unpack_single_arg=False
        )

    def train_map(
        self,
        stage: "PosteriorMapStage",
        chain: int,
        chain_total: int,
        *,
        savepath: "Path | None" = None,
    ) -> None:
        if savepath is not None:
            stage_savepath = savepath / "map" / f"{stage.fname}_{chain}"
            stage_savepath.mkdir(parents=True, exist_ok=True)
            if (stage_savepath / self.ckpt_path).exists():
                self.log.debug(
                    f"Loading MAP stage: {stage.name}_{chain} from {stage_savepath}"
                )
                self.load_checkpoint(stage_savepath, load_map=True)

                return
        self.log.debug(f"Running MAP stage: {stage.name}_{chain}")

        self.map.setup(stage, chain)

        results = tfp.optimizer.lbfgs_minimize(
            self.lbfgs,
            initial_position=self.map.position.current,
            tolerance=self.tolerance,
            x_tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            num_correction_pairs=1,
        )

        improved = results.objective_value < self.map.negative_log_prob
        self.map.improved.assign(improved)

        self.map.chain_min.assign(
            tf.where(
                improved,
                chain_total * tf.ones(self.sn_dim, dtype=tf.int32),
                self.map.chain_min,
            )
        )

        self.map.converged.assign(
            tf.where(
                improved,
                results.converged,
                self.map.converged,
            )
        )
        self.map.num_evaluations.assign_add(results.num_objective_evaluations)
        self.map.num_chain_evaluations.assign(results.num_objective_evaluations)
        self.map.negative_log_prob.assign(
            tf.where(
                improved,
                results.objective_value,
                self.map.negative_log_prob,
            )
        )

        ind = 0
        initial_position = []
        current_position = []
        if self.map.train_delta_m:
            initial_delta_m = self.map.position.current[:, ind : ind + 1]
            delta_m = results.position[:, ind : ind + 1]
            ind += 1
            initial_position.append(initial_delta_m)
            current_position.append(delta_m)
        else:
            initial_delta_m = self.map.delta_m.current
            delta_m = self.map.delta_m.current
        self.map.delta_m.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                initial_delta_m,
                self.map.delta_m.initial,
            )
        )
        self.map.delta_m.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                delta_m,
                self.map.delta_m.best,
            )
        )

        if self.map.train_delta_p:
            initial_delta_p = self.map.position.current[:, ind : ind + 1]
            delta_p = results.position[:, ind : ind + 1]
            ind += 1
            initial_position.append(initial_delta_p)
            current_position.append(delta_p)
        else:
            initial_delta_p = self.map.delta_p.current
            delta_p = self.map.delta_p.current
        self.map.delta_p.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                initial_delta_p,
                self.map.delta_p.initial,
            )
        )
        self.map.delta_p.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                delta_p,
                self.map.delta_p.best,
            )
        )

        if self.map.train_bias:
            initial_bias = self.map.position.current[:, ind : ind + 1]
            bias = results.position[:, ind : ind + 1]
            ind += 1
            initial_position.append(initial_bias)
            current_position.append(bias)
        else:
            initial_bias = self.map.bias.current
            bias = self.map.bias.current
        self.map.bias.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                bias,
                self.map.bias.initial,
            )
        )
        self.map.bias.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                bias,
                self.map.bias.best,
            )
        )

        if self.map.nflow.physical_latents:
            initial_u_delta_av = self.map.position.current[:, ind : ind + 1]
            u_delta_av = results.position[:, ind : ind + 1]
            ind += 1
            initial_position.append(initial_u_delta_av)
            current_position.append(u_delta_av)
        else:
            initial_u_delta_av = self.map.u_delta_av.current
            u_delta_av = self.map.u_delta_av.current
        self.map.u_delta_av.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                initial_u_delta_av,
                self.map.u_delta_av.initial,
            )
        )
        self.map.u_delta_av.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                u_delta_av,
                self.map.u_delta_av.best,
            )
        )

        initial_u_latents = self.map.position.current[:, ind:]
        u_latents = results.position[:, ind:]
        initial_position.append(initial_u_latents)
        current_position.append(u_latents)
        self.map.u_latents.initial.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=self.map.n_u_latents,
                    axis=-1,
                ),
                initial_u_latents,
                self.map.u_latents.initial,
            )
        )
        self.map.u_latents.best.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=self.map.n_u_latents,
                    axis=-1,
                ),
                u_latents,
                self.map.u_latents.best,
            )
        )

        if self.nflow.physical_latents:
            initial_us = tf.concat([initial_u_delta_av, initial_u_latents], axis=-1)
            us = tf.concat([u_delta_av, u_latents], axis=-1)
        else:
            initial_us = initial_u_latents
            us = u_latents

        initial_z_latents = self.nflow.u_to_z(initial_us, permute=True)
        z_latents = self.nflow.u_to_z(us, permute=True)

        if self.nflow.physical_latents:
            initial_delta_av = initial_z_latents[:, 0:1]
            initial_z_latents = initial_z_latents[:, 1:]
            delta_av = z_latents[:, 0:1]
            z_latents = z_latents[:, 1:]
        else:
            initial_delta_av = self.map.delta_av.current
            delta_av = self.map.delta_av.current

        self.map.z_latents.initial.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=self.map.n_z_latents,
                    axis=-1,
                ),
                initial_z_latents,
                self.map.z_latents.initial,
            )
        )
        self.map.z_latents.best.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=self.map.n_z_latents,
                    axis=-1,
                ),
                z_latents,
                self.map.z_latents.best,
            )
        )

        self.map.delta_av.initial.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                initial_delta_av,
                self.map.delta_av.initial,
            )
        )
        self.map.delta_av.best.assign(
            tf.where(
                tf.repeat(tf.expand_dims(improved, axis=-1), repeats=1, axis=-1),
                delta_av,
                self.map.delta_av.best,
            )
        )

        initial_position = tf.concat(initial_position, axis=-1)
        current_position = tf.concat(current_position, axis=-1)

        self.map.position.initial.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=initial_position.shape[-1],
                    axis=-1,
                ),
                initial_position,
                self.map.position.initial,
            )
        )
        self.map.position.best.assign(
            tf.where(
                tf.repeat(
                    tf.expand_dims(improved, axis=-1),
                    repeats=current_position.shape[-1],
                    axis=-1,
                ),
                current_position,
                self.map.position.best,
            )
        )

        if savepath is not None:
            self.log.debug(
                f"Saving MAP stage: {stage.name}_{chain} from {stage_savepath}"
            )
            (stage_savepath / self.ckpt_path).mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(stage_savepath, save_map=True)

    # === HMC Functions ===

    def hmc_train(
        self,
        *,
        savepath: "Path | None" = None,
    ) -> None:
        if savepath is not None:
            hmc_savepath = savepath / "hmc"
            hmc_savepath.mkdir(parents=True, exist_ok=True)
            if (hmc_savepath / self.ckpt_path).exists():
                self.log.debug(f"Loading HMC from {hmc_savepath}")
                self.load_checkpoint(hmc_savepath, load_hmc=True)

                samples = self.hmc.samples.numpy()

                ind = 0
                if self.map.train_delta_m:
                    delta_m = samples[..., ind : ind + 1]
                    ind += 1
                else:
                    delta_m = self.hmc.delta_m
                if self.map.train_delta_p:
                    delta_p = samples[..., ind : ind + 1]
                    ind += 1
                else:
                    delta_p = self.hmc.delta_p
                if self.map.nflow.physical_latents:
                    u_delta_av = samples[..., ind : ind + 1]
                    ind += 1
                else:
                    u_delta_av = self.hmc.u_delta_av
                u_latents = samples[..., ind:]
                if self.map.nflow.physical_latents:
                    us = np.concat([u_delta_av, u_latents], axis=-1)
                else:
                    us = u_latents
                us = us.reshape(-1, self.map.n_flow_latents)
                # Transform u_latents to z_latents
                z_latents = (
                    self.nflow.u_to_z(us, permute=True)
                    .numpy()
                    .reshape(*samples.shape[:-1], self.map.n_flow_latents)
                )
                if self.map.pae.physical_latents:
                    delta_av = z_latents[..., 0:1]
                    z_latents = z_latents[..., 1:]
                else:
                    delta_av = self.hmc.delta_av

                vars(self)["hmc"] = PosteriorHMCValue(
                    tf.Variable(samples),
                    tf.Variable(self.hmc.step_sizes_final),
                    tf.Variable(self.hmc.is_accepted),
                    tf.Variable(u_delta_av),
                    tf.Variable(u_latents),
                    tf.Variable(delta_av),
                    tf.Variable(z_latents),
                    tf.Variable(delta_m),
                    tf.Variable(delta_p),
                )

                return
        self.log.debug("Running HMC")

        initial_position = self.map.position.best

        mask = np.squeeze(self.sn_mask.astype(np.bool_), axis=(1, 2))

        pae_input = tf.concat((self.data.time, self.data.amplitude), axis=-1)
        z_latents = self.pae.encoder(
            pae_input,
            training=False,
            mask=self.data.mask,
        ).numpy()[mask][:, 0, :]
        z_latent_std = np.std(z_latents, axis=0)

        zs = z_latents[
            :, : self.map.pae.n_z_latents + (1 if self.map.pae.physical_latents else 0)
        ]
        if not self.nflow.physical_latents:
            zs = zs[:, 1:]

        u_latents = self.nflow.z_to_u(zs, permute=True)
        u_latent_std = np.std(u_latents, axis=0)

        stds = []
        ind = 0
        if self.map.train_delta_m:
            stds.append(z_latent_std[ind : ind + 1])
            ind += 1
        if self.map.train_delta_p:
            stds.append(z_latent_std[ind : ind + 1])
            ind += 1

        ind = 0
        if self.map.nflow.physical_latents:
            stds.append(u_latent_std[ind : ind + 1])
            ind += 1
        stds.append(u_latent_std[ind:])
        step_size = tf.zeros_like(initial_position) + tf.concat(stds, axis=-1)

        progress = tqdm(
            total=self.n_leapfrog * (self.n_burnin + self.n_samples),
            leave=True,
        )

        @tf.py_function(Tout=[])
        def update_progress() -> None:
            progress.update()

        def unnormalized_posterior_log_prob(pos):
            update_progress()
            return self(
                (
                    self.map.get_position(pos, best=True),
                    self.data.time,
                    self.data.amplitude,
                    self.data.sigma,
                ),
                training=False,
                mask=self.data.mask,
            )

        def trace_fn(_, pkr):
            step_size = pkr.inner_results.accepted_results.step_size
            is_accepted = pkr.inner_results.is_accepted
            return [step_size, is_accepted]

        def sample_chain():
            # run hmc
            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                num_leapfrog_steps=self.n_leapfrog,
                step_size=step_size,
            )

            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=hmc,
                num_adaptation_steps=int(self.n_burnin * 0.8),
                target_accept_prob=self.target_acceptance_rate,
            )

            samples, [step_sizes_final, is_accepted] = tfp.mcmc.sample_chain(
                self.n_samples,
                initial_position,
                kernel=kernel,
                num_burnin_steps=self.n_burnin,
                trace_fn=trace_fn,
                name="run",
            )

            return samples, step_sizes_final, is_accepted

        samples, step_sizes_final, is_accepted = sample_chain()
        samples, step_sizes_final, is_accepted = (
            samples.numpy(),
            step_sizes_final.numpy(),
            is_accepted.numpy(),
        )

        ind = 0
        if self.map.train_delta_m:
            delta_m = samples[..., ind : ind + 1]
            ind += 1
        else:
            delta_m = self.hmc.delta_m
        if self.map.train_delta_p:
            delta_p = samples[..., ind : ind + 1]
            ind += 1
        else:
            delta_p = self.hmc.delta_p
        if self.map.nflow.physical_latents:
            u_delta_av = samples[..., ind : ind + 1]
            ind += 1
        else:
            u_delta_av = self.hmc.u_delta_av
        u_latents = samples[..., ind:]
        if self.map.nflow.physical_latents:
            us = np.concat([u_delta_av, u_latents], axis=-1)
        else:
            us = u_latents
        us = us.reshape(-1, self.map.n_flow_latents)
        # Transform u_latents to z_latents
        z_latents = (
            self.nflow.u_to_z(us, permute=True)
            .numpy()
            .reshape(*samples.shape[:-1], self.map.n_flow_latents)
        )
        if self.map.pae.physical_latents:
            delta_av = z_latents[..., 0:1]
            z_latents = z_latents[..., 1:]
        else:
            delta_av = self.hmc.delta_av

        vars(self)["hmc"] = PosteriorHMCValue(
            tf.Variable(samples),
            tf.Variable(step_sizes_final),
            tf.Variable(is_accepted),
            tf.Variable(u_delta_av),
            tf.Variable(u_latents),
            tf.Variable(delta_av),
            tf.Variable(z_latents),
            tf.Variable(delta_m),
            tf.Variable(delta_p),
        )

        if savepath is not None:
            self.save_checkpoint(hmc_savepath, save_hmc=True)
