# Copyright 2025 Patrick Armstrong
import os
from typing import TYPE_CHECKING

import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow_probability import distributions as tfd

if TYPE_CHECKING:
    from numpy import typing as npt

    from supaernova.steps.pae.tf import TFPAEModel
    from supaernova.steps.nflow.tf import TFNFlowModel
    from supaernova.steps.posterior import TFPosteriorModel
    from supaernova.configs.steps.data import DataStepResult
    from supaernova.configs.steps.posterior import PosteriorMAPStage


class PosteriorMapValue(tf.Module):
    def __init__(self, initial: tf.Variable) -> None:
        self.original: tf.Variable = initial
        self.initial: tf.Variable = initial
        self.current: tf.Variable = initial
        self.best: tf.Variable = initial


class PosteriorMap(tf.Module):
    def __init__(
        self,
        config: "TFPosteriorModel",
        nflow: "TFNFlowModel",
        pae: "TFPAEModel",
        data: "DataStepResult",
    ) -> None:
        self.random_initial_positions: bool = config.options.random_initial_positions
        # Equivalent to `self.name = ...` but avoids tf / ks from tracking self.name
        vars(self)["nflow"]: TFNFlowModel = nflow
        vars(self)["pae"]: TFPAEModel = pae
        self.data: DataStepResult = data

        self.data_mask: npt.NDArray[bool] = config.data_mask
        self.sn_mask: npt.NDArray[bool] = config.sn_mask
        self.spec_mask: npt.NDArray[bool] = config.spec_mask
        self.wl_mask: npt.NDArray[bool] = config.wl_mask

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
        self.num_evaluations = tf.Variable(tf.constant(0), dtype=tf.int32)
        self.num_chain_evaluations = tf.Variable(tf.constant(0), dtype=tf.int32)
        self.negative_log_prob = tf.Variable(
            np.inf * tf.ones(self.sn_dim, dtype=tf.float32)
        )

        # === Priors ===
        self.u_delta_av_min: float = config.options.u_delta_av_min
        self.u_delta_av_max: float = config.options.u_delta_av_max
        self.u_delta_av_start: float = config.options.u_delta_av_start
        self.u_delta_av_end: float = config.options.u_delta_av_end
        self.u_delta_av_mean: float = config.options.u_delta_av_mean
        self.u_delta_av_std: float = config.options.u_delta_av_std
        self.u_delta_av_prior = tfd.Normal(
            loc=self.u_delta_av_mean, scale=self.u_delta_av_std
        )
        if self.nflow.physical_latents:
            self.n_pos += 1

        self.u_latents_min: float = config.options.u_latents_min
        self.u_latents_max: float = config.options.u_latents_max
        self.u_latents_mean: float = config.options.u_latents_mean
        self.u_latents_std: float = config.options.u_latents_std
        self.u_latents_prior = tfd.MultivariateNormalDiag(
            loc=self.u_latents_mean * tf.ones(self.n_u_latents),
            scale_diag=self.u_latents_std * tf.ones(self.n_u_latents),
        )

        self.delta_av_min: float = config.options.delta_av_min
        self.delta_av_max: float = config.options.delta_av_max
        self.delta_av_start: float = config.options.delta_av_start
        self.delta_av_end: float = config.options.delta_av_end
        self.delta_av_mean: float = config.options.delta_av_mean
        self.delta_av_std: float = config.options.delta_av_std
        self.delta_av_prior = tfd.Normal(
            loc=self.delta_av_mean, scale=self.delta_av_std
        )

        self.train_delta_m: bool = config.options.train_delta_m
        self.delta_m_min: float = config.options.delta_m_min
        self.delta_m_max: float = config.options.delta_m_max
        self.delta_m_start: float = config.options.delta_m_start
        self.delta_m_end: float = config.options.delta_m_end
        self.delta_m_mean: float = config.options.delta_m_mean
        self.delta_m_std: float = config.options.delta_m_std
        self.delta_m_prior = tfd.Normal(loc=self.delta_m_mean, scale=self.delta_m_std)
        if self.train_delta_m:
            self.n_pos += 1

        self.train_delta_p: bool = config.options.train_delta_p
        self.delta_p_min: float = config.options.delta_p_min
        self.delta_p_max: float = config.options.delta_p_max
        self.delta_p_start: float = config.options.delta_p_start
        self.delta_p_end: float = config.options.delta_p_end
        self.delta_p_mean: float = config.options.delta_p_mean
        self.delta_p_std: float = config.options.delta_p_std
        self.delta_p_prior = tfd.Normal(loc=self.delta_p_mean, scale=self.delta_p_std)
        if self.train_delta_p:
            self.n_pos += 1

        self.train_bias: bool = config.options.train_bias
        self.bias_min: float = config.options.bias_min
        self.bias_max: float = config.options.bias_max
        self.bias_start: float = config.options.bias_start
        self.bias_end: float = config.options.bias_end
        self.bias_mean: float = config.options.bias_mean
        self.bias_std: float = config.options.bias_std
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
        stage: "PosteriorMAPStage",
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
                        self.u_delta_av_end - self.u_delta_av_start
                    ) / stage.n_chains
                    u_delta_av_scale = (
                        self.u_delta_av_start
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
                z_latents = self.pae(
                    pae_input,
                    training=False,
                    mask=self.data_mask,
                    sn_mask=self.sn_mask,
                    spec_mask=self.spec_mask,
                    wl_mask=self.wl_mask,
                )[0][:, 0, :]
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
                            self.u_delta_av_end - self.u_delta_av_start
                        ) / stage.n_chains
                        u_delta_av_scale = (
                            self.u_delta_av_start
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
                            self.delta_av_end - self.delta_av_start
                        ) / stage.n_chains
                        delta_av_scale = (
                            self.delta_av_start
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
                    self.delta_av_end - self.delta_av_start
                ) / stage.n_chains
                delta_av_scale = (
                    self.delta_av_start + (stage.n_chains - chain) * delta_av_slope
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
            delta_m_slope = (self.delta_m_end - self.delta_m_start) / stage.n_chains
            delta_m_scale = (
                self.delta_m_start + (stage.n_chains - chain) * delta_m_slope
            )
            delta_m = tf.zeros((self.sn_dim, 1)) + delta_m_scale
        elif stage.init_delta_m == "constant":
            delta_m = self.delta_m_mean * tf.ones((self.sn_dim, 1))

        # --- delta_p ---
        if stage.init_delta_p == "random":
            delta_p = self.delta_p_prior.sample((self.sn_dim, 1))
        elif stage.init_delta_p == "scale":
            delta_p_slope = (self.delta_p_end - self.delta_p_start) / stage.n_chains
            delta_p_scale = (
                self.delta_p_start + (stage.n_chains - chain) * delta_p_slope
            )
            delta_p = tf.zeros((self.sn_dim, 1)) + delta_p_scale
        elif stage.init_delta_p == "constant":
            delta_p = self.delta_p_mean * tf.ones((self.sn_dim, 1))

        # --- bias ---
        if stage.init_bias == "random":
            bias = self.bias_prior.sample((self.sn_dim, 1))
        elif stage.init_bias in {"scale", "constant"}:
            bias = self.bias_mean * tf.ones((self.sn_dim, 1))

        delta_m = tf.clip_by_value(delta_m, self.delta_m_min, self.delta_m_max)
        delta_p = tf.clip_by_value(delta_p, self.delta_p_min, self.delta_p_max)
        bias = tf.clip_by_value(bias, self.bias_min, self.bias_max)
        u_delta_av = tf.clip_by_value(
            u_delta_av, self.u_delta_av_min, self.u_delta_av_max
        )
        u_latents = tf.clip_by_value(u_latents, self.u_latents_min, self.u_latents_max)

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

        delta_m = tf.clip_by_value(delta_m, self.delta_m_min, self.delta_m_max)
        delta_p = tf.clip_by_value(delta_p, self.delta_p_min, self.delta_p_max)
        bias = tf.clip_by_value(bias, self.bias_min, self.bias_max)
        u_delta_av = tf.clip_by_value(
            u_delta_av, self.u_delta_av_min, self.u_delta_av_max
        )
        u_latents = tf.clip_by_value(u_latents, self.u_latents_min, self.u_latents_max)

        return tf.concat([delta_m, delta_p, bias, u_delta_av, u_latents], axis=-1)

    def prior(self, position) -> tf.Tensor:
        log_prior = tf.zeros((position.shape[0],))
        inf_prior = -tf.ones_like(log_prior) * np.inf

        delta_m = position[:, 0:1]
        delta_p = position[:, 1:2]
        bias = position[:, 2:3]
        u_delta_av = position[:, 3:4]
        u_latents = position[:, 4:]

        u_latents_log_prior = self.u_latents_prior.log_prob(u_latents)
        log_prior += tf.where(
            tf.math.is_nan(u_latents_log_prior), inf_prior, u_latents_log_prior
        )

        if self.train_delta_m:
            delta_m_log_prior = self.delta_m_prior.log_prob(delta_m)[:, 0]
            log_prior += tf.where(
                tf.math.is_nan(delta_m_log_prior), inf_prior, delta_m_log_prior
            )

        if self.train_delta_p:
            delta_p_log_prior = self.delta_p_prior.log_prob(delta_p)[:, 0]
            log_prior += tf.where(
                tf.math.is_nan(delta_p_log_prior), inf_prior, delta_p_log_prior
            )

        if self.train_bias:
            bias_log_prior = self.bias_prior.log_prob(bias)[:, 0]
            log_prior += tf.where(
                tf.math.is_nan(bias_log_prior), inf_prior, bias_log_prior
            )

        if self.nflow.physical_latents:
            u_delta_av_log_prior = self.u_delta_av_prior.log_prob(u_delta_av)[:, 0]
            log_prior += tf.where(
                tf.math.is_nan(u_delta_av_log_prior), inf_prior, u_delta_av_log_prior
            )

        return log_prior
