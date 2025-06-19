import os
from typing import Annotated

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

type FTensor[Shape: tuple[int | str]] = Annotated[tf.Tensor, tf.float32, Shape]
type ITensor[Shape: tuple[int | str]] = Annotated[tf.Tensor, tf.int32, Shape]
type FRTensor[Shape: tuple[int | str]] = Annotated[tf.RaggedTensor, tf.float32, Shape]
type IRTensor[Shape: tuple[int | str]] = Annotated[tf.RaggedTensor, tf.int32, Shape]
type Tensor[Shape: tuple[int | str]] = FTensor[Shape] | ITensor[Shape]
type RaggedTensor[Shape: tuple[int | str]] = FRTensor[Shape] | IRTensor[Shape]
type GenericTensor[Shape: tuple[int | str]] = Tensor[Shape] | RaggedTensor[Shape]
