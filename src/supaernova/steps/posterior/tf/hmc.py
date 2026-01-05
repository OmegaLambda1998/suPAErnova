# Copyright 2025 Patrick Armstrong
from supaernova._tf import tf


class PosteriorHMCValue(tf.Module):
    def __init__(
        self,
        samples: tf.Variable,
        # step_sizes_final: tf.Variable,
        # is_accepted: tf.Variable,
        log_prior: tf.Variable,
        log_like: tf.Variable,
        log_prob: tf.Variable,
        zs: tf.Variable,
        # u_delta_av: tf.Variable,
        # u_latents: tf.Variable,
        # delta_av: tf.Variable,
        # z_latents: tf.Variable,
        # delta_m: tf.Variable,
        # delta_p: tf.Variable,
    ) -> None:
        self.samples: tf.Variable = samples
        # self.step_sizes_final: tf.Variable = step_sizes_final
        # self.is_accepted: tf.Variable = is_accepted
        self.log_prior: tf.Variable = log_prior
        self.log_like: tf.Variable = log_like
        self.log_prob: tf.Variable = log_prob
        self.zs: tf.Variable = zs

        # self.u_delta_av: tf.Variable = u_delta_av
        # self.u_latents: tf.Variable = u_latents
        # self.delta_av: tf.Variable = delta_av
        # self.z_latents: tf.Variable = z_latents
        # self.delta_m: tf.Variable = delta_m
        # self.delta_p: tf.Variable = delta_p
