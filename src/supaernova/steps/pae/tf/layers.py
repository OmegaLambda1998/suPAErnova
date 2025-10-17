from typing import TYPE_CHECKING

from supaernova._tf import TF_CTX, IS_ROCM, ks, tf

if TYPE_CHECKING:
    from typing import Any, Self
    from collections.abc import Callable

    type Dense = ks.layers.Dense | ROCMDense


def ROCMLambda(fn):
    def ROCM_fn(*args, **kwargs):
        with TF_CTX:
            return fn(*args, **kwargs)

    return ROCM_fn


@ks.utils.register_keras_serializable("SuPAErnova")
class ROCMDense(ks.layers.Layer):
    def __init__(
        self: "Self",
        units: int,
        *args,
        trainable: bool = True,
        activation: "str | Callable[[tf.Tensor], tf.Tensor] | dict[str, Any] | None" = None,
        use_bias: bool = True,
        kernel_initializer: ks.initializers.Initializer | str | None = "glorot_uniform",
        kernel_regularizer: ks.regularizers.Regularizer | str | None = None,
        kernel_constraint: ks.constraints.Constraint | str | None = None,
        bias_initializer: ks.initializers.Initializer | str | NameError = "zeros",
        dtype: tf.DType = tf.float32,
        **kwargs: "Any",
    ) -> None:
        super().__init__(*args, **kwargs)
        self.units: int = units
        self.trainable = trainable
        self.activation: ks.activations._Activation = (
            ks.activations.get(activation)
            if isinstance(activation, str)
            else activation
        )
        self.use_bias: bool = use_bias
        self.kernel_initializer: ks.initializers.Initializer = (
            ks.initializers.get(kernel_initializer)
            if isinstance(kernel_initializer, str)
            else kernel_initializer
        )
        self.kernel_regularizer: ks.regularizers.Regularizer = (
            ks.regularizers.get(kernel_regularizer)
            if isinstance(kernel_regularizer, str)
            else kernel_regularizer
        )
        self.kernel_constraint: ks.constraints.Constraint = (
            ks.constraints.get(kernel_constraint)
            if isinstance(kernel_constraint, str)
            else kernel_constraint
        )
        self.bias_initializer: ks.initializers.Initializer = (
            ks.initializers.get(bias_initializer)
            if isinstance(bias_initializer, str)
            else bias_initializer
        )

        self.kernel: tf.Variable
        self.bias: tf.Variable | None

    def build(self: "Self", input_shape: tuple[int, ...]) -> None:
        input_dim = int(input_shape[-1])
        with TF_CTX:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=self.trainable,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=self.trainable,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self: "Self", inputs: tf.Tensor):
        # Standard matmul
        output = tf.matmul(inputs, self.kernel)
        # Safe bias addition using broadcasting
        if self.use_bias:
            output += self.bias
        # Activation
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self: "Self"):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": ks.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": ks.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": ks.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": ks.constraints.serialize(self.kernel_constraint),
            "bias_initializer": ks.initializers.serialize(self.bias_initializer),
            "dtype": self.dtype.name,
        })
        return config


# ROCM Has some issues with different kernels. Hopefully will improve with time
# In the meantime, these layers and functions work around those issues
DENSE: type["Dense"] = ROCMDense if IS_ROCM else ks.layers.Dense
REDUCE_SUM: "Callable" = (
    ROCMLambda(tf.math.reduce_sum) if IS_ROCM else tf.math.reduce_sum
)
REDUCE_MEAN: "Callable" = (
    ROCMLambda(tf.math.reduce_mean) if IS_ROCM else tf.math.reduce_mean
)
REDUCE_ANY: "Callable" = (
    ROCMLambda(tf.math.reduce_any) if IS_ROCM else tf.math.reduce_any
)
REDUCE_ALL: "Callable" = (
    ROCMLambda(tf.math.reduce_all) if IS_ROCM else tf.math.reduce_all
)
