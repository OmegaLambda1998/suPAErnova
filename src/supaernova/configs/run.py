# Copyright 2025 Patrick Armstrong
from time import time
from functools import cached_property

from pydantic import computed_field, model_validator

from supaernova.steps import Step
from supaernova.steps.pae import PAEStep
from supaernova.steps.data import DataStep
from supaernova.steps.nflow import NFlowStep
from supaernova.steps.sim import SimStep
from supaernova.steps.posterior import PosteriorStep

from .input import InputConfig
from .steps import StepConfig
from .steps.pae import PAEStepConfig
from .steps.data import DataStepConfig
from .steps.nflow import NFlowStepConfig
from .steps.sim import SimStepConfig
from .steps.posterior import PosteriorStepConfig


class RunConfig(InputConfig):
    # === Class Variables ===
    # === Class Methods ===
    # === Field Variables ===
    # --- Required ---
    # --- Optional ---
    # Configs
    data: DataStepConfig | None = None
    pae: PAEStepConfig | None = None
    nflow: NFlowStepConfig | None = None
    sim: SimStepConfig | None = None
    posterior: PosteriorStepConfig | None = None

    @computed_field
    @cached_property
    def step_configs(self) -> list[StepConfig]:
        return [
            step_config
            for step_config in [
                self.data,
                self.pae,
                self.nflow,
                self.sim,
                self.posterior,
            ]
            if step_config is not None
        ]

    # Steps
    data_step: DataStep | None = None
    pae_step: PAEStep | None = None
    nflow_step: NFlowStep | None = None
    sim_step: SimStep | None = None
    posterior_step: PosteriorStep | None = None

    @computed_field
    @cached_property
    def steps(self) -> list[Step]:
        return [Step.steps[step.id](step) for step in self.step_configs]

    # === Model Validators ===
    # --- Before ---
    # --- After ---
    @model_validator(mode="after")
    def _validate_steps(self):
        if len(self.step_configs) == 0:
            err = f"No steps have been defined! Please specify at least one of {list(Step.steps.keys())}"
            self._raise(err)

        for step_config in self.step_configs:
            for required_step in step_config.required_steps:
                if getattr(self, required_step) is None:
                    err = f"{step_config.id} requires that {required_step} is run first, but {required_step} has not been defined!"
                    self._raise(err)
                step = getattr(self, required_step)
                step_variants = step.variants[:]
                variants = step_config.variants
                if len(step.proxies.get(step.id, [])) > 0:
                    for proxy in step.proxies[step.id]:
                        proxy_step = getattr(self, proxy.id)
                        if proxy_step is not None:
                            proxy_variants = proxy_step.variants
                            step_variants += proxy_variants
                for i, variant in enumerate(variants):
                    if not hasattr(variant, required_step):
                        setattr(variant, required_step, i)
                    ind = getattr(variant, required_step)
                    if isinstance(ind, str) and ind not in [
                        v.name for v in step_variants
                    ]:
                        err = f"{step_config.id} requires a {required_step}.{ind} variant, but only {[f'{required_step}.{v.name}' for v in step_variants]} variants have been defined!"
                        self._raise(err)
                    if isinstance(ind, int) and len(step_variants) <= ind:
                        err = f"{step_config.id}.{variant.name} requires that at least {ind + 1} {required_step} variants exist, but only {len(step_variants)} variants have been defined!"
                        self._raise(err)
        return self

    # === Field Validators ===
    # --- Before ---
    # --- After ---
    # === Instance Methods ===
    def require(self, step_name: str, *, is_proxy: bool = False) -> Step:
        step = getattr(self, step_name + "_step")
        if step is None and not is_proxy:
            err = f"{step_name} has not yet run"
            self._raise(err)
        return step

    def requirements(self, required_steps: list[str]) -> dict[str, Step]:
        steps = {}
        for required_step in required_steps:
            steps[required_step] = self.require(required_step)
            required = getattr(self, required_step)
            steps = {**self.requirements(required.required_steps), **steps}
            if len(required.proxies.get(required.id, [])) > 0:
                for proxy_step in required.proxies[required.id]:
                    proxy_require = self.require(proxy_step.id, is_proxy=True)
                    if proxy_require is not None:
                        steps[proxy_step.id] = proxy_require
        return steps

    def run(self) -> None:
        for step in self.steps:
            if not step.skip:
                self.log.info(f"Executing {step.name}")
                start_time = time()
                args = []
                kwargs = self.requirements(step.options.required_steps)
                step.analyse(*args, **kwargs)
                setattr(self, step.id + "_step", step)
                step.clear(*args, **kwargs)
                end_time = time()
                exec_time = end_time - start_time
                unit = "s"
                t = 60
                if exec_time > t:
                    exec_time /= t
                    unit = "m"
                if exec_time > t:
                    exec_time /= t
                    unit = "h"
                self.log.info(f"{step.name} took {exec_time:.2f}{unit}")

    # === Static Methods ===
