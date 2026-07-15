from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supaernova.configs.steps.data import SNPAEData, LazySNPAEData
    from numpy import typing as npt
    from collections.abc import Callable


class Filter:
    def __init__(self, path: Path):
        self.name = path.stem
        lines = path.read_text(encoding="utf8").strip().split("\n")
        self.zp = float(lines[0])
        self.wavelength = np.array([
            float(line.strip().split()[0]) for line in lines[1:]
        ])
        self.throughput = np.array([
            float(line.strip().split()[1]) for line in lines[1:]
        ])
        self.effective_wavelength = self.wavelength[np.argmax(self.throughput)]
