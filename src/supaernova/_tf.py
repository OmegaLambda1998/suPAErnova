import os

NPROC = str(os.cpu_count())

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Number of CPUs available
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# Number of CPU cores available
os.environ["TF_NUM_INTRAOP_THREADS"] = NPROC
os.environ["OMP_NUM_THREADS"] = NPROC
os.environ["MKL_NUM_THREADS"] = NPROC

from contextlib import nullcontext

import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_probability as tfp
from tensorflow_probability import (
    bijectors as tfb,
    distributions as tfd,
)

GPUS = tf.config.list_physical_devices("GPU")
tf.config.set_soft_device_placement(True)
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.threading.set_inter_op_parallelism_threads(int(NPROC))
tf.config.threading.set_intra_op_parallelism_threads(int(NPROC))

IS_GPU = len(GPUS) > 0
IS_ROCM = any(
    "AMD" in tf.config.experimental.get_device_details(gpu).get("device_name", "")
    for gpu in GPUS
)
TF_CTX = tf.device("/CPU:0") if IS_ROCM else nullcontext()
JIT_COMPILE = IS_GPU

HUGE = tf.float16.max
