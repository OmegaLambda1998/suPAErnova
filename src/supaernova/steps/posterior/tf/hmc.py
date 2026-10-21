# Copyright 2025 Patrick Armstrong
from supaernova._tf import tf


class PosteriorHMCValue(tf.Module):
    def __init__(
        self,
        samples: tf.Variable,
        log_prior: tf.Variable,
        log_like: tf.Variable,
        log_prob: tf.Variable,
        zs: tf.Variable,
    ) -> None:
        self.samples: tf.Variable = samples
        self.log_prior: tf.Variable = log_prior
        self.log_like: tf.Variable = log_like
        self.log_prob: tf.Variable = log_prob
        self.zs: tf.Variable = zs
