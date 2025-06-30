= Data

#figure(image(
  "../../tests/cache/paper_parity/data/snpae/DataStepConfig/plots/12345/summary.svg",
))

= Probabilistic AutoEncoder

== Training Variables: $Delta A_v$

#figure(grid(
  columns: 2,
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/1/latents.svg",
  ),
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/1/residual.svg",
  ),
))


== Training Variables: $Delta A_v$, $z_1$, $z_2$, $z_3$

#figure(grid(
  columns: 2,
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/4/latents.svg",
  ),
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/4/residual.svg",
  ),
))

== Training Variables: $Delta A_v$, $z_1$, $z_2$, $z_3$, $Delta_cal(M)$

#figure(grid(
  columns: 2,
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/5/latents.svg",
  ),
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/5/residual.svg",
  ),
))

== Training Variables: $Delta A_v$, $z_1$, $z_2$, $z_3$, $Delta_cal(M)$, $Delta_cal(p)$

#figure(grid(
  columns: 2,
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/6/latents.svg",
  ),
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/6/residual.svg",
  ),
))

== Final summary

#figure(grid(
  columns: 2,
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/latents.svg",
  ),
  image(
    "../../tests/cache/paper_parity/pae/snpae/12345/PAEStepConfig/TFPAEModelConfig/plots/12345/residual.svg",
  ),
))

= Normalising Flow

== $z#sub[latents]$

#figure(image(
  "../../tests/cache/paper_parity/nflow/snpae/12345/NFlowStepConfig/TFNFlowModelConfig/plots/12345/z_latents.svg",
))

== $mu#sub[latents]$

#figure(image(
  "../../tests/cache/paper_parity/nflow/snpae/12345/NFlowStepConfig/TFNFlowModelConfig/plots/12345/u_latents.svg",
))

== Latents

#figure(image(
  "../../tests/cache/paper_parity/nflow/snpae/12345/NFlowStepConfig/TFNFlowModelConfig/plots/12345/latents.svg",
))

== Normalising Flow Animation

#figure(
  image(
    "../../tests/cache/paper_parity/nflow/snpae/12345/NFlowStepConfig/TFNFlowModelConfig/plots/12345/steps/latent_steps.gif",
  ),
)

= Posterior Analysis

== MAP
#figure(
  grid(
    columns: 2,
    image(
      "../../tests/cache/paper_parity/posterior/snpae/12345/PosteriorStepConfig/TFPosteriorModelConfig/plots/12345/train/map_best.svg",
    ),
    image(
      "../../tests/cache/paper_parity/posterior/snpae/12345/PosteriorStepConfig/TFPosteriorModelConfig/plots/12345/test/map_best.svg",
    ),
  ),
)

== HMC

#figure(
  grid(
    columns: 2,
    image(
      "../../tests/cache/paper_parity/posterior/snpae/12345/PosteriorStepConfig/TFPosteriorModelConfig/plots/12345/train/hmc.svg",
    ),
    image(
      "../../tests/cache/paper_parity/posterior/snpae/12345/PosteriorStepConfig/TFPosteriorModelConfig/plots/12345/test/hmc.svg",
    ),
  ),
)

== Dispersion

#figure(
  grid(
    columns: 2,
    image(
      "../../tests/cache/paper_parity/posterior/snpae/12345/PosteriorStepConfig/TFPosteriorModelConfig/plots/12345/train/dispersion.svg",
    ),
    image(
      "../../tests/cache/paper_parity/posterior/snpae/12345/PosteriorStepConfig/TFPosteriorModelConfig/plots/12345/test/dispersion.svg",
    ),
  ),
)
