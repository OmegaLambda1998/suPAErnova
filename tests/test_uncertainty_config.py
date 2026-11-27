"""The shared uncertainty config: presets, overrides, legacy options and validation."""

import pytest

from supaernova.configs.uncertainty import (
    PRESETS,
    UncertaintyModel,
    check_consistent,
    pop_legacy_fields,
)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets_validate(preset: str) -> None:
    model = UncertaintyModel(preset=preset)
    assert model.preset == preset


def test_preset_values() -> None:
    main = UncertaintyModel(preset="main")
    assert (main.calibration.scale, main.calibration.floor) == (1.4, 4e-10)
    assert main.calibration.stage == "before_bands"
    assert main.correlation.mode == "independent"
    assert main.correlation.band_coherence == "coherent"
    assert main.model_error.scale == "model_and_noise"
    assert not main.sim.inject

    dev = UncertaintyModel(preset="dev")
    assert (dev.calibration.scale, dev.calibration.floor) == (1.0, 0.0)
    assert dev.correlation.mode == "epoch"
    assert dev.model_error.estimator == "ratio_of_sums"

    assert UncertaintyModel(preset="proposal").has_auto()
    assert not UncertaintyModel(preset="proposal_fixed").has_auto()


def test_override_deep_merges() -> None:
    model = UncertaintyModel(preset="dev", correlation={"rho": 0.4})
    assert model.correlation.rho == 0.4
    assert model.correlation.mode == "epoch"  # kept from the preset


def test_legacy_fields_map_to_dev() -> None:
    data = pop_legacy_fields(
        {"sigma_scale": 1.4, "fractional_error": False, "recon_correlation": 0.2, "x": 1},
        None,
    )
    assert data["x"] == 1
    model = UncertaintyModel(**data["uncertainty"])
    assert model.preset == "dev"
    assert model.calibration.scale == 1.4
    assert model.model_error.scale == "constant"
    assert model.correlation.rho == 0.2


def test_legacy_with_table_raises() -> None:
    def fail(msg: str) -> None:
        raise ValueError(msg)

    with pytest.raises(ValueError, match="cannot be combined"):
        pop_legacy_fields({"sigma_scale": 1.4, "uncertainty": {"preset": "main"}}, fail)


@pytest.mark.parametrize(
    "bad",
    [
        {"preset": "dev", "correlation": {"mode": "kernel"}},  # auto params need kernel_ml
        {"preset": "proposal", "calibration": {"scale": 1.4}},  # a would be applied twice
        {"preset": "main", "correlation": {"rho": 0.3}},  # rho is epoch-only
        {"preset": "dev", "model_error": {"estimator": "mean_ratio"}},
        {"preset": "proposal_fixed", "kernel": {"model": {"wavelength": {"w0": 0.5}}}},
    ],
)
def test_invalid_combinations_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        UncertaintyModel(**bad)


def test_consistency_check() -> None:
    dev = UncertaintyModel(preset="dev")
    check_consistent(dev, None, what="old result")  # results without a model pass
    check_consistent(dev, dev.model_dump(), what="same")
    rho = UncertaintyModel(preset="dev", correlation={"rho": 0.5})
    check_consistent(rho, dev.model_dump(), what="model error may differ")
    with pytest.raises(ValueError, match="calibration"):
        check_consistent(UncertaintyModel(preset="main"), dev.model_dump(), what="data")
