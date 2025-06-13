# SuPAErnova

This repository contains the codes required to the train models and perform analyses for

*A Probabilistic Autoencoder for Type Ia Supernovae Spectral Time Series*

Constructed in TensorFlow 2 and TensorFlow Probability.

## Terminology

In an attempt to maintain consistency throughout this codebase I've chosen the following terms as standards for variable names, functions, etc…

### [Dimensions](./src/suPAErnova/typing/dimensions.py)

- `sn_dim: SNDim`: Traverses across each SN in a given batch / survey.
- `spec_dim: SpecDim`: Traverse across each spectrum of a given SN.
- `wl_dim: WLDim`: Traverse across each wavelength of a given spectrum.
- `phase_dim: PhaseDim`: The value `1`, tracks “the number of phases for a given spectrum” which is always `1`.

### [Data](./src/suPAErnova/typing/steps/data.py)

### [PAE](./src/suPAErnova/typing/steps/pae/pae.py)

- `n_physical_latents: NPhysicalLatents`: The number of **physical** PAE latents, always either `0` or `3`.
- `n_z_latents: NZLatents`: The number of **non-physical** PAE latents.
- `n_pae_latents: NPAELatents`: The **total** number of PAE latents.

#### [TensorFlow](./src/suPAErnova/typing/steps/pae/tf.py)

### [NFlow](./src/suPAErnova/typing/steps/nflow.py)

### [Posterior](./src/suPAErnova/typing/steps/posterior.py)
