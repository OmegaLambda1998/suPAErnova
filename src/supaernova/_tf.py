import os

NPROC = str(os.cpu_count())

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# "tensorpool" is not a recognised TF_GPU_ALLOCATOR value (only "BFC" and
# "cuda_malloc_async" are); it silently falls back to the default allocator.
# Pin it explicitly to that default so CUDA and ROCm take the same path.
os.environ["TF_GPU_ALLOCATOR"] = "BFC"

# Number of CPUs available
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# Number of CPU cores available
os.environ["TF_NUM_INTRAOP_THREADS"] = NPROC
os.environ["OMP_NUM_THREADS"] = NPROC
os.environ["MKL_NUM_THREADS"] = NPROC

NPROC = int(NPROC)

import gc
from contextlib import nullcontext

import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_probability as tfp
from tensorflow_probability import (
    bijectors as tfb,
    distributions as tfd,
)

ks.utils.set_random_seed(1)

# Opt-in fast-math mode. Off by default: op-determinism + TF32-disabled keep
# CUDA and ROCm bit-comparable and every run reproducible. Set
# SNPAE_FAST_MATH=1 to trade that for speed on a single backend -- re-enables
# nondeterministic (faster) reduction/atomics kernels and, on NVIDIA Ampere+,
# TF32 fp32 matmul/conv. Results then vary run-to-run and backend-to-backend.
FAST_MATH = os.environ.get("SNPAE_FAST_MATH", "0").lower() in {"1", "true", "yes"}

if not FAST_MATH:
    tf.config.experimental.enable_op_determinism()

# TF32 is an NVIDIA-only reduced-precision math mode (Ampere and newer) that
# TensorFlow enables by default for fp32 matmul/conv. It has no ROCm
# equivalent, so leaving it on makes CUDA runs systematically less precise
# than ROCm runs for the exact same "float32" model/gradients. Disable it
# (unless fast-math is requested) so both backends compute fp32 ops at full
# precision.
tf.config.experimental.enable_tensor_float_32_execution(FAST_MATH)

GPUS = tf.config.list_physical_devices("GPU")
tf.config.set_soft_device_placement(True)
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.threading.set_inter_op_parallelism_threads(NPROC)
tf.config.threading.set_intra_op_parallelism_threads(NPROC)

tf.config.optimizer.set_experimental_options({
    "layout_optimizer": False,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_model_pruning": False,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": False,
    "implementation_selector": True,
    "auto_mixed_precision": False,
    "disable_meta_optimizer": False,
    "min_graph_nodes": False,
    "auto_parallel": False,
})

print(tf.config.optimizer.get_experimental_options())

IS_GPU = len(GPUS) > 0
IS_ROCM = any(
    "AMD" in tf.config.experimental.get_device_details(gpu).get("device_name", "")
    for gpu in GPUS
)
TF_CTX = tf.device("/CPU:0") if IS_ROCM else nullcontext()
JIT_COMPILE = False

HUGE = tf.float16.max


def mem_trace() -> dict[str, str]:
    trace: dict[str, str] = {}
    if IS_GPU:
        trace = {
            k: f"{(v * 1e-9):.2f}GB"
            for k, v in tf.config.experimental.get_memory_info("GPU:0").items()
        }
    return trace


def clear_session() -> None:
    ks.backend.clear_session()
    gc.collect()
