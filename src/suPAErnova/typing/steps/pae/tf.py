from typing import Literal

import numpy as np
from numpy import typing as npt

from suPAErnova.typing.dimensions import SNDim, WLDim, SpecDim, PhaseDim
from suPAErnova.typing.backends.tf import FTensor, ITensor, IRTensor

from .pae import NZLatents, NPAELatents

# --- Data Shapes ---
type PhaseShape = tuple[SNDim, SpecDim, PhaseDim]
type DPhaseShape = tuple[SNDim, SpecDim, PhaseDim]
type AmpShape = tuple[SNDim, SpecDim, WLDim]
type SigmaShape = tuple[SNDim, SpecDim, WLDim]
type MaskShape = tuple[SNDim, SpecDim, WLDim]
type PhysicalLatentsShape = tuple[SNDim, SpecDim, Literal[1]]
type ZLatentsShape = tuple[SNDim, SpecDim, NZLatents]
type PAELatentsShape = tuple[SNDim, SpecDim, NPAELatents]

# --- Data Tensors ---
type PhaseTensor = FTensor[PhaseShape]
type DPhaseTensor = FTensor[DPhaseShape]
type AmpTensor = FTensor[AmpShape]
type SigmaTensor = FTensor[SigmaShape]
type MaskTensor = ITensor[MaskShape]
type PhysicalLatentsTensor = FTensor[PhysicalLatentsShape]
type ZLatentsTensor = FTensor[ZLatentsShape]
type PAELatentsTensor = FTensor[PAELatentsShape]

# --- Encoder Tensors ---
type EncoderInputsShape = tuple[SNDim, SpecDim, Literal["PhaseDim + WLDim"]]

type EncoderInputs = FTensor[*EncoderInputsShape]
type EncoderOutputs = PAELatentsTensor

# --- Decoder Tensors ---
type DecoderInputsShape = tuple[SNDim, SpecDim, Literal["PhaseDim + NPAELatents"]]

type DecoderInputs = FTensor[*DecoderInputsShape]
type DecoderOutputs = AmpTensor

# --- Model Tensors ---
type ModelInputs = tuple[PhaseTensor, AmpTensor, SigmaTensor, MaskTensor]

type EpochInputs = tuple[
    tuple[
        PhaseTensor,
        DPhaseTensor,
        AmpTensor,
        SigmaTensor,
        MaskTensor,
        MaskTensor,
        MaskTensor,
    ],
]
